//! Pure metadata contract. No model, estimate or numeric Q can manufacture proof.
use super::*;
pub(super) const CAPACITY:usize=64;
pub(super) const ADMISSION_LIMIT:usize=8;

#[derive(Clone,Copy,Debug,Default,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Bits {pub raw:bool,pub coupled:bool}
impl Bits {
    pub fn any(self)->bool {self.raw||self.coupled}
    pub fn retained(self,next:Self)->bool {(!self.raw||next.raw)&&(!self.coupled||next.coupled)}
}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Entry {
    pub key:String,
    pub proof_sha256:String,
    pub support_sha256:String,
    pub acquired_actor:String,
    pub bits:Bits,
    pub epoch:u64,
}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Registry {
    schema:String,
    pub anchor:String,
    pub beta_bits:u64,
    pub epoch:u64,
    pub admitted_this_epoch:usize,
    pub entries:Vec<Entry>,
}
impl Registry {
    pub fn new(anchor:String,beta:f64)->Result<Self> {
        if anchor.is_empty()||!beta.is_finite() {return Err(invalid("invalid registry anchor/beta"));}
        Ok(Self {schema:"diagnostic-proof-registry-v1".into(),anchor,beta_bits:beta.to_bits(),epoch:0,admitted_this_epoch:0,entries:vec![]})
    }
    pub fn begin_epoch(&mut self,epoch:u64)->Result<()> {
        if epoch<=self.epoch {return Err(invalid("registry epoch cannot reset admission budget"));}
        self.epoch=epoch;self.admitted_this_epoch=0;Ok(())
    }
    pub fn admit(&mut self,entry:Entry)->Result<bool> {
        if entry.epoch!=self.epoch||self.epoch==0||entry.acquired_actor!=self.anchor
            ||entry.key.is_empty()||entry.proof_sha256.is_empty()||entry.support_sha256.is_empty() {
            return Err(invalid("registry admission lacks verified identity/current accepted actor"));
        }
        if !entry.bits.any() {return Ok(false);}
        if self.entries.iter().any(|e|e.key==entry.key) {return Err(invalid("duplicate registry witness"));}
        if self.admitted_this_epoch>=ADMISSION_LIMIT||self.entries.len()>=CAPACITY {
            return Err(invalid("registry admission or capacity budget exhausted"));
        }
        self.entries.push(entry);self.admitted_this_epoch+=1;Ok(true)
    }
    // Explicit retirement, not a failed candidate clearing its own obligation.
    pub fn retire(&mut self,key:&str)->Result<Entry> {
        let i=self.entries.iter().position(|e|e.key==key).ok_or_else(||invalid("unknown retirement"))?;
        Ok(self.entries.remove(i))
    }
    pub fn checkpoint(&self)->Result<serde_json::Value> {
        let payload=serde_json::to_value(self)?;
        Ok(serde_json::json!({"sha256":sha256(&serde_json::to_vec(&payload)?),"payload":payload}))
    }
    pub fn restore(saved:&serde_json::Value,anchor:&str,beta:f64)->Result<Self> {
        let payload=&saved["payload"];
        if saved["sha256"].as_str()!=Some(sha256(&serde_json::to_vec(payload)?).as_str()) {
            return Err(invalid("registry checkpoint digest mismatch"));
        }
        let out:Self=serde_json::from_value(payload.clone())?;
        if out.schema!="diagnostic-proof-registry-v1"||out.anchor!=anchor||out.beta_bits!=beta.to_bits()
            ||out.entries.len()>CAPACITY||out.admitted_this_epoch>ADMISSION_LIMIT
            ||out.entries.iter().any(|e|!e.bits.any()||e.acquired_actor!=out.anchor||e.epoch>out.epoch)
            ||out.entries.iter().map(|e|&e.key).collect::<std::collections::BTreeSet<_>>().len()!=out.entries.len() {
            return Err(invalid("registry checkpoint contract differs"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(i:usize,epoch:u64)->Entry {Entry {key:format!("proof-{i}"),proof_sha256:"verified-file".into(),support_sha256:"verified-support".into(),acquired_actor:"accepted".into(),bits:Bits {raw:false,coupled:true},epoch}}
    #[test] fn dynamic_registry_bits_are_independent_and_do_not_require_an_unknown_success() {
        assert!(Bits {raw:false,coupled:true}.retained(Bits {raw:false,coupled:true}));
        assert!(!Bits {raw:false,coupled:true}.retained(Bits {raw:true,coupled:false}));
        assert!(Bits::default().retained(Bits::default()));
        let mut r=Registry::new("accepted".into(),16.).unwrap();r.begin_epoch(1).unwrap();
        let mut e=entry(0,1);e.bits=Bits::default();assert!(!r.admit(e).unwrap());assert!(r.entries.is_empty());
    }
    #[test] fn dynamic_registry_eight_admissions_cannot_be_reset_within_an_epoch() {
        let mut r=Registry::new("accepted".into(),16.).unwrap();r.begin_epoch(1).unwrap();
        for i in 0..8 {assert!(r.admit(entry(i,1)).unwrap());}
        assert!(r.admit(entry(8,1)).is_err());assert!(r.begin_epoch(1).is_err());
        let old=r.retire("proof-0").unwrap();assert!(old.bits.coupled);
        assert!(r.admit(entry(8,1)).is_err()); // Retirement does not replenish admission budget.
        r.begin_epoch(2).unwrap();assert!(r.admit(entry(8,2)).unwrap());
    }
    #[test] fn dynamic_registry_capacity_and_explicit_retirement_keep_order() {
        let mut r=Registry::new("accepted".into(),16.).unwrap();
        for epoch in 1..=8 {r.begin_epoch(epoch).unwrap();for i in 0..8 {r.admit(entry((epoch as usize-1)*8+i,epoch)).unwrap();}}
        r.begin_epoch(9).unwrap();assert!(r.admit(entry(64,9)).is_err());
        r.retire("proof-3").unwrap();r.admit(entry(64,9)).unwrap();
        assert_eq!(r.entries.len(),64);assert_eq!(r.entries[3].key,"proof-4");assert_eq!(r.entries.last().unwrap().key,"proof-64");
    }
    #[test] fn dynamic_registry_resume_preserves_next_admission_and_rejects_changed_anchor_or_bits() {
        let mut r=Registry::new("accepted".into(),16.).unwrap();r.begin_epoch(1).unwrap();r.admit(entry(0,1)).unwrap();
        let saved=r.checkpoint().unwrap();let mut resumed=Registry::restore(&saved,"accepted",16.).unwrap();
        r.admit(entry(1,1)).unwrap();resumed.admit(entry(1,1)).unwrap();assert_eq!(r,resumed);
        assert!(Registry::restore(&saved,"other",16.).is_err());assert!(Registry::restore(&saved,"accepted",8.).is_err());
        let mut bad=saved;bad["payload"]["entries"][0]["bits"]["coupled"]=serde_json::json!(false);
        assert!(Registry::restore(&bad,"accepted",16.).is_err());
    }
}
