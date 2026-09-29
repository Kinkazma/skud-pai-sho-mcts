//! Isolated f64 value-head cost study. No model schema or production update.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, hint::black_box, io::Read, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
#[derive(Clone)]
struct Layer { inputs: usize, outputs: usize, weights: Vec<f64>, biases: Vec<f64> }
#[derive(Clone)]
struct Head { layers: Vec<Layer> }
struct Workspace { activations: Vec<Vec<f64>>, deltas: Vec<Vec<f64>>, gradients: Vec<Layer> }
struct Case { state: Vec<f64>, target: f64, actions: Vec<[f64;32]>, source: u64, training: Option<MicroExample> }
impl Head {
    fn random(widths: &[usize], seed: u64) -> Self {
        let mut rng=StableRng::new(seed);
        Self { layers:widths.windows(2).map(|w| {
            let scale=(6.0/(w[0]+w[1]) as f64).sqrt();
            Layer {inputs:w[0],outputs:w[1],weights:(0..w[0]*w[1]).map(|_|(rng.next_f64()*2.-1.)*scale).collect(),biases:vec![0.;w[1]]}
        }).collect() }
    }
    fn current(model: &MicroModel) -> Self {
        assert!(model.has_spatial());
        let w=model.parameters();let mut head=Self::random(&[417,32,1],0);
        for j in 0..32 {
            head.layers[0].weights[j*417..j*417+128].copy_from_slice(&w[MICRO_VALUE_TRUNK+j*128..MICRO_VALUE_TRUNK+(j+1)*128]);
            head.layers[0].weights[j*417+128..(j+1)*417].copy_from_slice(&w[MICRO_VALUE_BOARD+j*289..MICRO_VALUE_BOARD+(j+1)*289]);
            head.layers[0].biases[j]=w[MICRO_VALUE_TRUNK+4096+j];
        }
        head.layers[1].weights.copy_from_slice(&w[4128..4160]);
        head.layers[1].biases[0]=w[4160];head
    }
    fn parameters(&self) -> usize {self.layers.iter().map(|l|l.weights.len()+l.biases.len()).sum()}
    fn workspace(&self) -> Workspace {
        Workspace {activations:self.layers.iter().map(|l|vec![0.;l.outputs]).collect(),
            deltas:self.layers.iter().map(|l|vec![0.;l.outputs]).collect(),
            gradients:self.layers.iter().map(|l|Layer {inputs:l.inputs,outputs:l.outputs,weights:vec![0.;l.weights.len()],biases:vec![0.;l.outputs]}).collect()}
    }
    // This matches production's dense prefix + occupied-board sum exactly.
    // The scan is charged once per value call for every architecture.
    fn forward(&self, state: &[f64], ws: &mut Workspace) -> f64 {
        let mut occupied=[0usize;289];let mut count=0;
        for (i,x) in state[128..].iter().enumerate() {if *x!=0. {occupied[count]=i;count+=1;}}
        let mut final_raw=0.;
        for (index,layer) in self.layers.iter().enumerate() {
            let (previous,next)=ws.activations.split_at_mut(index);
            let input=if index==0 {state} else {&previous[index-1]};
            for (j,out) in next[0].iter_mut().enumerate() {
                let w=&layer.weights[j*layer.inputs..(j+1)*layer.inputs];
                let raw=if index==0 {
                    let base=layer.biases[j]+input[..128].iter().zip(&w[..128]).map(|(x,w)|x*w).sum::<f64>();
                    base+occupied[..count].iter().map(|i|input[128+i]*w[128+i]).sum::<f64>()
                } else {layer.biases[j]+input.iter().zip(w).map(|(x,w)|x*w).sum::<f64>()};
                *out=raw.tanh();final_raw=raw;
            }
        }
        final_raw
    }
    fn gradient(&self, state: &[f64], target: f64, ws: &mut Workspace) -> f64 {
        self.forward(state,ws);
        let last=self.layers.len()-1;let value=ws.activations[last][0];
        for d in &mut ws.deltas {d.fill(0.);}
        ws.deltas[last][0]=(value-target)*(1.-value*value);
        for index in (0..self.layers.len()).rev() {
            let l=&self.layers[index];let input=if index==0 {state} else {&ws.activations[index-1]};
            for j in 0..l.outputs {
                let d=ws.deltas[index][j];ws.gradients[index].biases[j]=d;
                for (i,x) in input.iter().enumerate() {
                    ws.gradients[index].weights[j*l.inputs+i]=d*x;
                    if index>0 {ws.deltas[index-1][i]+=d*l.weights[j*l.inputs+i];}
                }
            }
            if index>0 {for (d,h) in ws.deltas[index-1].iter_mut().zip(input) {*d*=1.-h*h;}}
        }
        0.5*(value-target).powi(2)
    }
    fn finite_differences(&self, cases: &[Case]) -> Result<Value> {
        let mut probe=self.clone();let mut ws=self.workspace();let mut max_error=0.0_f64;let mut checked=0;
        let mut rng=StableRng::new(51219);
        let dense=Case {state:(0..417).map(|_|rng.next_f64()*0.5-0.25).collect(),target:0.3,actions:vec![],source:0,training:None};
        let checks:Vec<_>=cases.iter().step_by((cases.len()/3).max(1)).take(3).chain(std::iter::once(&dense)).collect();
        for case in checks {
            self.gradient(&case.state,case.target,&mut ws);
            for layer in 0..self.layers.len() {
                for bias in [false,true] {
                    let source=if bias {&self.layers[layer].biases} else {&self.layers[layer].weights};
                    let n=source.len().min(24);
                    for k in 0..n {
                        let i=if n==source.len() {k} else {rng.index(source.len())};
                        let expected=if bias {ws.gradients[layer].biases[i]} else {ws.gradients[layer].weights[i]};
                        let mut losses=[0.;2];
                        for (t,sign) in [-1.,1.].iter().enumerate() {
                            let p=if bias {&mut probe.layers[layer].biases[i]} else {&mut probe.layers[layer].weights[i]};
                            *p=source[i]+sign*1e-5;
                            let mut scratch=probe.workspace();let v=probe.forward(&case.state,&mut scratch).tanh();
                            losses[t]=0.5*(v-case.target).powi(2);
                        }
                        if bias {probe.layers[layer].biases[i]=source[i];} else {probe.layers[layer].weights[i]=source[i];}
                        let actual=(losses[1]-losses[0])/2e-5;let error=(expected-actual).abs();
                        max_error=max_error.max(error);checked+=1;
                        if error>1e-7+1e-5*expected.abs() {return Err(format!("gradient mismatch {layer}/{i}: {expected} vs {actual}").into());}
                    }
                }
            }
        }
        Ok(json!({"checked":checked,"max_absolute_error":max_error}))
    }
}
fn load(path: &Path) -> Result<Vec<Case>> {
    let manifest:Value=serde_json::from_slice(&fs::read(path)?)?;
    let mut out=vec![];
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let raw=fs::read(row["targets"].as_str().ok_or("targets")?)?;
        if hash(&raw)!=row["targets_sha256"] {return Err("target hash mismatch".into());}
        let mut bytes=vec![];flate2::read::GzDecoder::new(raw.as_slice()).read_to_end(&mut bytes)?;
        let saved:Vec<SavedMicroExample>=serde_json::from_slice(&bytes)?;
        let e=&saved[row["index"].as_u64().ok_or("index")? as usize];
        let bytes=fs::read(row["psr"].as_str().ok_or("psr")?)?;
        if hash(&bytes)!=row["psr_sha256"] {return Err("PSR hash mismatch".into());}
        let record:GameRecord=std::str::from_utf8(&bytes)?.parse()?;record.replay()?;
        let mut p=record.initial_position();for a in record.actions().iter().take(e.decision-1) {p.apply(*a)?;}
        let state=micro_spatial_state_features(&p);
        if !state.iter().zip(&e.state).all(|(a,b)|a.to_bits()==b.to_bits()) || e.state.len()!=417 {return Err("spatial feature mismatch".into());}
        let example=e.example_for_rules(record.rules())?;
        out.push(Case {state,target:e.value,actions:example.actions.clone(),source:example.sequence_source,training:Some(example)});
    }
    Ok(out)
}
fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if !(5..=6).contains(&args.len()) {return Err("usage: micro_value_depth_probe MODEL MANIFEST OUTPUT ROUNDS ORDER [overlay]".into());}
    let overlay=args.get(5).is_some_and(|x|x=="overlay");
    if args.len()==6 && !overlay {return Err("unknown mode".into());}
    let rounds:usize=args[3].parse()?;if rounds==0 {return Err("positive rounds required".into());}
    let artifact=MicroArtifact::load(Path::new(&args[0]))?;let model=artifact.model()?;
    let cases=load(Path::new(&args[1]))?;let current=Head::current(&model);
    let mut ws=current.workspace();
    for c in &cases {if current.forward(&c.state,&mut ws).tanh().to_bits()!=model.embed(&c.state).value.to_bits() {return Err("extracted current value mismatch".into());}}
    let mut heads=vec![("417-32-1",current.clone()),("417-64-32-1",Head::random(&[417,64,32,1],9473)),("417-128-64-32-1",Head::random(&[417,128,64,32,1],9473))];
    let mut neutral=heads[1].1.clone();let last=neutral.layers.last_mut().unwrap();last.weights.fill(0.);last.biases.fill(0.);
    let mut nws=neutral.workspace();let neutral_exact=cases.iter().all(|c| {
        let old=current.forward(&c.state,&mut ws);let delta=neutral.forward(&c.state,&mut nws);
        (old+delta).tanh().to_bits()==old.tanh().to_bits()
    });
    // A literal identity matrix between tanh layers is not a neutral upgrade.
    let mut inserted=Head::random(&[417,64,32,1],0);
    for l in &mut inserted.layers {l.weights.fill(0.);l.biases.fill(0.);}
    inserted.layers[0].weights[..32*417].copy_from_slice(&current.layers[0].weights);
    inserted.layers[0].biases[..32].copy_from_slice(&current.layers[0].biases);
    for i in 0..32 {inserted.layers[1].weights[i*64+i]=1.;}
    inserted.layers[2]=current.layers[1].clone();let mut iws=inserted.workspace();
    let mut changed=0;let mut difference=0.0_f64;
    for c in &cases {
        let a=current.forward(&c.state,&mut ws).tanh();let b=inserted.forward(&c.state,&mut iws).tanh();
        changed+=usize::from(a.to_bits()!=b.to_bits());difference=difference.max((a-b).abs());
    }
    if args[4]=="reverse" {heads.reverse();} else if args[4]!="forward" {return Err("invalid order".into());}
    let mut results=vec![];
    for (name,head) in &heads {
        let gradient_check=head.finite_differences(&cases)?;let mut ws=head.workspace();
        if overlay {
            for c in &cases {model.memory_context(&c.state,c.source)?;}
            let start=Instant::now();let mut checksum=0.;
            for _ in 0..rounds {for c in &cases {
                let embedding=model.embed(black_box(&c.state));
                let extra=if *name=="417-32-1" {0.} else {head.forward(&c.state,&mut ws)};
                let logits=MicroModel::logits(&embedding,&c.actions);
                let priors=if logits.is_empty() {vec![]} else {model.memory_priors(&c.state,&c.actions,&micro_softmax(&logits)?,c.source)?};
                black_box(priors);checksum+=black_box((embedding.value,extra)).0;
            }}
            let seconds=start.elapsed().as_secs_f64();
            let gradient_rounds=(rounds/5).max(1);let start=Instant::now();
            for _ in 0..gradient_rounds {for c in &cases {
                black_box(model.loss_gradient(c.training.as_ref().unwrap())?);
                if *name!="417-32-1" {
                    black_box(head.gradient(&c.state,c.target,&mut ws));
                    black_box(&ws.gradients);
                }
            }}
            let gradient_seconds=start.elapsed().as_secs_f64();
            results.push(json!({"name":name,"calls":rounds*cases.len(),"augmented_forward_seconds":seconds,
                "gradient_calls":gradient_rounds*cases.len(),"augmented_gradient_seconds":gradient_seconds,
                "checksum":checksum,"gradient_check":gradient_check,"additional_parameters":if *name=="417-32-1" {0} else {head.parameters()}}));
            eprintln!("{name}: full inference + extra head {seconds:.3}s; gradient {gradient_seconds:.3}s");
            continue;
        }
        for c in &cases {black_box(head.gradient(&c.state,c.target,&mut ws));}
        let start=Instant::now();let mut checksum=0.;
        for _ in 0..rounds {for c in &cases {
            head.forward(black_box(&c.state),&mut ws);
            checksum+=black_box(ws.activations.last().unwrap()[0]);
        }}
        let forward_seconds=start.elapsed().as_secs_f64();
        let gradient_rounds=(rounds/10).max(1);let start=Instant::now();
        for _ in 0..gradient_rounds {for c in &cases {checksum+=black_box(head.gradient(black_box(&c.state),c.target,&mut ws));black_box(&ws.gradients);}}
        let gradient_seconds=start.elapsed().as_secs_f64();
        let parameters=head.parameters();
        results.push(json!({"name":name,"parameters":parameters,"inference_calls":rounds*cases.len(),"gradient_calls":gradient_rounds*cases.len(),
            "forward_seconds":forward_seconds,"gradient_seconds":gradient_seconds,"checksum":checksum,"gradient_check":gradient_check,
            "whole_model_parameters_if_replaced":model.parameters().len()-current.parameters()+parameters,
            "whole_model_parameters_if_added_as_residual":model.parameters().len()+parameters}));
        eprintln!("{name}: forward {forward_seconds:.3}s; gradient {gradient_seconds:.3}s");
    }
    fs::write(&args[2],serde_json::to_vec_pretty(&json!({"model":artifact.identity(),"cases":cases.len(),"order":args[4],"rounds":rounds,
        "current_value_exact_cases":cases.len(),"neutral_residual_exact":neutral_exact,
        "identity_layer_insertion":{"changed_cases":changed,"max_absolute_value_change":difference},"results":results,
        "scope":if overlay {"Actual frozen value/policy/retrieval forward and gradient plus independent active-head computation. No joint loss, value combination, search, learning, replacement or strength measurement."}
            else {"Reused-workspace scalar f64 value kernels; no search, policy, minibatch optimizer, production replacement or strength measurement."}}))?)?;
    Ok(())
}
