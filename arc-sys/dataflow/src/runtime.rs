use std::sync::Arc;
use std::sync::Mutex;

pub struct Runtime {
    #[cfg(feature = "thread-pinning")]
    topology: Arc<Mutex<Topology>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "thread-pinning")]
            topology: Arc::new(Mutex::new(Topology::new())),
            threads: Vec::new(),
        }
    }

    pub fn spawn<Fut>(&mut self, instance: Fut, cpu_id: usize) -> &mut Self
    where
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        #[cfg(feature = "thread-pinning")]
        let topology = self.topology.clone();
        self.threads.push(std::thread::spawn(move || {
            #[cfg(feature = "thread-pinning")]
            bind_thread_to_cpu(cpu_id, topology);
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(tokio::task::LocalSet::new().run_until(instance));
        }));
        self
    }
}

#[cfg(feature = "thread-pinning")]
fn bind_thread_to_cpu(cpu_id: usize, topology: Arc<Mutex<Topology>>) {
    cfg_if::cfg_if! {
        if #[cfg(target_os = "linux")] {
            let thread_id = unsafe { libc::pthread_self() };
            let mut topology = topology.lock().unwrap();
            let cpus = topology.objects_with_type(&ObjectType::Core).unwrap();
            let cpuset = cpus.get(cpu_id).expect("Core not found").cpuset().unwrap();
            topology
                .set_cpubind_for_thread(thread_id, cpuset, CPUBIND_THREAD)
                .unwrap();
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        for thread in self.threads.drain(..) {
            thread.join().expect("Failed to join thread");
        }
    }
}
