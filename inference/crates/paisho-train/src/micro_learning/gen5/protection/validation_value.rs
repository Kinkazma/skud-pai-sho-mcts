//! Alignment with the secondary Guard's balanced value MSE.
//! It adds one aggregate constraint, never one hard constraint per witness.
use super::*;

pub(super) struct ValidationValue {
    pub rows: Vec<Arc<MicroExample>>,
    pub counts: [usize; 3],
    pub anchor_loss: f64,
}
impl ValidationValue {
    pub fn new(rows: Vec<Arc<MicroExample>>, model: &MicroModel, parallel: Option<&cpu::Ordered>) -> Result<Self> {
        if rows.is_empty() || rows.len() > 512 || !model.has_deep_value() {
            return Err(invalid("validation-value constraint requires 1..512 rows and separated deep value"));
        }
        let mut counts = [0; 3];
        let rows = rows.into_iter().map(|row| {
            if ![-1., 0., 1.].contains(&row.value) {
                return Err(invalid("validation-value constraint requires proved class labels"));
            }
            counts[(row.value as i8 + 1) as usize] += 1;
            // Only state and the proved value survive this conversion. Cloning
            // whole action tables then clearing them would retain their capacity.
            Ok(Arc::new(MicroExample {
                state: row.state.clone(), value: row.value,
                sequence_source: row.sequence_source,
                actions: Vec::new(), policy: Vec::new(), action_values: Vec::new(),
                policy_weight: 0., value_weight: 1., policy_support: false,
            }))
        }).collect::<Result<Vec<_>>>()?;
        let mut result = Self {rows, counts, anchor_loss: 0.};
        result.anchor_loss = result.loss(model, parallel)?;
        Ok(result)
    }
    pub fn loss(&self, model: &MicroModel, parallel: Option<&cpu::Ordered>) -> Result<f64> {
        let model = model.clone();
        let f = move |row: &Arc<MicroExample>| {
            ((row.value as i8 + 1) as usize, (model.value(&row.state) - row.value).powi(2))
        };
        let parts: Vec<_> = match parallel {
            Some(p) => p.map_owned(self.rows.clone(), |_| 1, f),
            None => self.rows.iter().map(f).collect(),
        };
        let mut errors = [0.; 3];
        for (class, error) in parts { errors[class] += error; }
        // Keep the same class order, within-class sum and final average as Guard.
        let mse = errors.iter().zip(self.counts).filter(|(_,n)| *n > 0)
            .map(|(e,n)| e / n as f64).sum::<f64>()
            / self.counts.iter().filter(|n| **n > 0).count() as f64;
        if !mse.is_finite() {return Err(invalid("non-finite validation-value constraint MSE"));}
        Ok(mse)
    }
    pub fn gradient(&self, model: &MicroModel, parallel: Option<&cpu::Ordered>) -> Result<Vec<f64>> {
        let mut sums = vec![vec![0.; model.parameters().len()]; 3];
        let model = model.clone();
        let f = move |row: &Arc<MicroExample>| {
            model.loss_gradient(row).map(|(_,g)| ((row.value as i8 + 1) as usize, g))
        };
        // Same bounded gradient concurrency as the existing four references.
        for chunk in self.rows.chunks(32) {
            let parts: Vec<_> = match parallel {
                Some(p) => p.map_owned(chunk.to_vec(), |_| 1, f.clone()),
                None => chunk.iter().map(&f).collect(),
            };
            for part in parts {
                let (class, gradient) = part.map_err(invalid)?;
                for (sum, g) in sums[class].iter_mut().zip(gradient) { *sum += g; }
            }
        }
        let classes = self.counts.iter().filter(|n| **n > 0).count() as f64;
        let mut gradient = vec![0.; sums[0].len()];
        for (sum, count) in sums.iter().zip(self.counts).filter(|(_,n)| *n > 0) {
            // Native value loss is HALF squared error; Guard uses full MSE.
            for (g, s) in gradient.iter_mut().zip(sum) { *g += 2. * (s / count as f64) / classes; }
        }
        if gradient.iter().any(|g| !g.is_finite()) {return Err(invalid("non-finite validation-value constraint gradient"));}
        Ok(gradient)
    }
    pub fn accepts(&self, loss: f64) -> bool {
        loss.is_finite() && loss <= self.anchor_loss + FINITE_LOSS_TOLERANCE
    }
    pub fn merit(&self, loss: f64) -> f64 {
        ((loss - (self.anchor_loss + FINITE_LOSS_TOLERANCE)).max(0.)
            / (self.anchor_loss.abs() + FINITE_LOSS_TOLERANCE)).powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use paisho_ai::{MICRO_DEEP_VALUE_START, MICRO_INPUTS, MICRO_HIDDEN, MICRO_VALUE_TRUNK, MICRO_NEURAL_MEMORY_START};
    fn row(value: f64, offset: f64) -> Arc<MicroExample> {
        Arc::new(MicroExample {state:(0..417).map(|i| ((i as f64 + offset) * 0.17).sin() * 0.03).collect(),
            actions:vec![],policy:vec![],action_values:vec![],policy_support:false,
            policy_weight:0.,value_weight:1.,value,sequence_source:0})
    }
    fn model() -> MicroModel {
        let initial = MicroModel::seeded(42).with_deep_value(314);
        let mut w = initial.parameters().to_vec(); let n = w.len();
        for (i, v) in w[n-33..].iter_mut().enumerate() { *v = (i as f64 + 1.).sin() * 0.1; }
        MicroModel::from_parameters(w).unwrap()
    }
    #[test]
    fn diagnostic_validation_value_balances_classes_not_rows() {
        let model = model(); let loser = row(-1.,0.); let winner = row(1.,3.);
        let balanced = ValidationValue::new(vec![loser.clone(),winner.clone()],&model,None).unwrap();
        let unbalanced = ValidationValue::new(vec![loser.clone(),loser.clone(),loser.clone(),loser,winner],&model,None).unwrap();
        assert!((balanced.loss(&model,None).unwrap()-unbalanced.loss(&model,None).unwrap()).abs()<2e-14);
        let a = balanced.gradient(&model,None).unwrap(); let b = unbalanced.gradient(&model,None).unwrap();
        for (a,b) in a.iter().zip(&b) {assert!((a-b).abs()<2e-14);}
        let expected = 0.5 * ((model.value(&balanced.rows[0].state)+1.).powi(2)
            + (model.value(&balanced.rows[1].state)-1.).powi(2));
        assert_eq!(balanced.anchor_loss.to_bits(),expected.to_bits());
    }
    #[test]
    fn diagnostic_validation_value_conversion_preserves_math_without_action_capacity() {
        let model=model();
        let mut source=row(1.,2.).as_ref().clone();
        source.actions=vec![[0.;32];17];source.policy=vec![1./17.;17];
        source.action_values=vec![Some(0.25);17];source.policy_support=true;
        source.policy_weight=1.;source.value_weight=0.;source.sequence_source=123;
        let source=Arc::new(source);
        let mut old=source.as_ref().clone();
        old.actions.clear();old.policy.clear();old.action_values.clear();
        old.policy_weight=0.;old.value_weight=1.;old.policy_support=false;
        let panel=ValidationValue::new(vec![source.clone()],&model,None).unwrap();
        let new=&panel.rows[0];
        assert_eq!((new.actions.capacity(),new.policy.capacity(),new.action_values.capacity()),(0,0,0));
        assert_eq!(new.sequence_source,old.sequence_source);
        assert_eq!(new.value.to_bits(),old.value.to_bits());
        assert!(new.state.iter().zip(&old.state).all(|(a,b)|a.to_bits()==b.to_bits()));
        assert_eq!(source.actions.len(),17);assert_eq!(source.action_values.len(),17);
        let (a,ga)=model.loss_gradient(&old).unwrap();let (b,gb)=model.loss_gradient(new).unwrap();
        assert_eq!((a.value.to_bits(),a.policy.to_bits()),(b.value.to_bits(),b.policy.to_bits()));
        assert!(ga.iter().zip(gb).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
    #[test]
    fn diagnostic_validation_value_gradient_matches_four_finite_coordinates_and_mask() {
        let model = model(); let rows=vec![row(-1.,0.),row(-1.,2.),row(0.,5.),row(1.,9.)];
        let panel=ValidationValue::new(rows,&model,None).unwrap();
        let g=panel.gradient(&model,None).unwrap();let n=g.len();
        let head=(MICRO_INPUTS+1)*MICRO_HIDDEN;
        for i in [head,MICRO_DEEP_VALUE_START,n-33,n-1] {
            let eps=1e-5; let mut plus=model.parameters().to_vec();let mut minus=plus.clone();
            plus[i]+=eps;minus[i]-=eps;
            let numeric=(panel.loss(&MicroModel::from_parameters(plus).unwrap(),None).unwrap()
                -panel.loss(&MicroModel::from_parameters(minus).unwrap(),None).unwrap())/(2.*eps);
            assert!((g[i]-numeric).abs()<3e-8,"coordinate {i}: {} != {numeric}",g[i]);
        }
        for (i,g) in g.iter().enumerate() {
            let value=(head..=head+MICRO_HIDDEN).contains(&i)
                || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i);
            if !value { assert_eq!(*g,0.,"policy coefficient {i} must stay zero"); }
        }
    }
    #[test]
    fn validation_value_adds_exactly_one_constraint() {
        let model=model();let rows=vec![row(-1.,0.),row(0.,3.),row(1.,6.)];
        let mut protection=Protection::new(&model,rows.clone()).unwrap();
        assert_eq!(protection.gradients.len(),4);
        assert!(protection.validation_value.is_none());
        assert!(protection.diagnostic_enable_validation_value(rows.clone()).is_err());
        protection.enable_loop_v3();protection.diagnostic_enable_validation_value(rows).unwrap();
        assert_eq!(protection.gradients.len(),5);assert_eq!(protection.gram.len(),5);
        assert_eq!(protection.value_gram.len(),4);
        protection.restore(&serde_json::Value::Null).unwrap();
        let extra=protection.validation_value.as_ref().unwrap();
        assert!(extra.accepts(extra.anchor_loss+0.5*FINITE_LOSS_TOLERANCE));
        assert!(!extra.accepts(extra.anchor_loss+2.*FINITE_LOSS_TOLERANCE));
        assert_eq!(extra.merit(extra.anchor_loss+0.5*FINITE_LOSS_TOLERANCE),0.);
    }
    #[test]
    fn validation_value_checkpoint_restores_constraints_and_next_update_exactly() {
        let model=model();let rows=vec![row(-1.,0.),row(0.,3.),row(1.,6.)];
        let dir=std::env::temp_dir().join(format!("paisho-value-panel-resume-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&dir).unwrap();
        let mut a=Protection::new(&model,rows.clone()).unwrap();a.enable_loop_v3();
        a.enable_validation_value(rows.clone()).unwrap();a.observe(&rows);
        let mut trained=model.clone();a.train_shared(&mut trained,&rows,0.01,0.).unwrap();
        a.checkpoint(&dir).unwrap();
        let saved=serde_json::json!({"checkpoint":a.saved});
        let mut b=Protection::new(&trained,rows.clone()).unwrap();b.enable_loop_v3();
        b.enable_validation_value(rows.clone()).unwrap();b.restore(&saved).unwrap();
        assert_eq!(a.validation_checkpoint_identity().unwrap(),b.validation_checkpoint_identity().unwrap());
        assert_eq!(a.updates,b.updates);assert_eq!(a.fresh.len(),b.fresh.len());
        assert_eq!(a.validation_value.as_ref().unwrap().anchor_loss.to_bits(),b.validation_value.as_ref().unwrap().anchor_loss.to_bits());
        for (a,b) in a.gradients.iter().flatten().zip(b.gradients.iter().flatten()) {assert_eq!(a.to_bits(),b.to_bits());}
        let mut left=trained.clone();let mut right=trained.clone();
        a.train_shared(&mut left,&rows,0.01,0.).unwrap();b.train_shared(&mut right,&rows,0.01,0.).unwrap();
        assert!(left.parameters().iter().zip(right.parameters()).all(|(a,b)|a.to_bits()==b.to_bits()));
        let mut missing=Protection::new(&model,rows.clone()).unwrap();missing.enable_loop_v3();
        assert!(missing.restore(&saved).is_err());
        let mut changed=Protection::new(&model,rows.clone()).unwrap();changed.enable_loop_v3();
        changed.enable_validation_value(vec![row(-1.,9.),row(0.,3.),row(1.,6.)]).unwrap();
        assert!(changed.restore(&saved).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn old_checkpoint_can_add_value_panel_without_changing_weights_or_presentations() {
        let model=model();let rows=vec![row(-1.,0.),row(0.,3.),row(1.,6.)];
        let dir=std::env::temp_dir().join(format!("paisho-value-panel-migration-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&dir).unwrap();
        let mut old=Protection::new(&model,rows.clone()).unwrap();old.enable_loop_v3();old.observe(&rows);
        old.updates=19;old.checkpoint(&dir).unwrap();
        let mut new=Protection::new(&model,rows.clone()).unwrap();new.enable_loop_v3();
        new.enable_validation_value(rows.clone()).unwrap();
        new.restore(&serde_json::json!({"checkpoint":old.saved})).unwrap();
        assert_eq!(new.updates,19);assert_eq!(new.fresh.len(),3);assert_eq!(new.gradients.len(),5);
        assert!(new.anchor.parameters().iter().zip(model.parameters()).all(|(a,b)|a.to_bits()==b.to_bits()));
        fs::remove_dir_all(dir).unwrap();
    }
}
