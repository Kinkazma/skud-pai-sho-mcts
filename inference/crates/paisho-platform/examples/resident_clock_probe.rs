//! Isolated clock/RAM probe; never opens a training model or campaign.
use paisho_platform::training_time as clock;
use std::{io::{self,Write},path::Path,time::{Duration,Instant}};
fn main()->io::Result<()> {
    let args:Vec<_>=std::env::args().collect();
    clock::enable(Path::new(&args[1]))?;
    let memory:Vec<u64>=(0..1_000_000).map(|n|n ^ 0x12345678).collect();
    let address=memory.as_ptr() as usize;
    let checksum:u64=memory.iter().sum();
    let wall=Instant::now();let started=clock::now();let end=started+Duration::from_secs(2);
    println!("{{\"pid\":{},\"address\":{},\"checksum\":{}}}",std::process::id(),address,checksum);
    io::stdout().flush()?;
    let mut ticks=0;
    while clock::now()<end { ticks+=1;std::thread::sleep(Duration::from_millis(10)); }
    assert_eq!(memory.as_ptr() as usize,address);assert_eq!(memory.iter().sum::<u64>(),checksum);
    println!("{{\"same_ram\":true,\"ticks\":{},\"active_seconds\":{},\"wall_seconds\":{},\"pause_seconds\":{}}}",ticks,clock::elapsed(started).as_secs_f64(),wall.elapsed().as_secs_f64(),clock::paused().as_secs_f64());
    Ok(())
}
