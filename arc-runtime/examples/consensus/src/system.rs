use kompact::prelude::*;

use kompact::config_keys::system::LABEL;
use kompact::config_keys::system::THREADS;
use kompact::executors::crossbeam_workstealing_pool;

use omnipaxos::leader_election::ballot_leader_election::messages::BLEMessage;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatMsg;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatReply;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatRequest;
use omnipaxos::leader_election::ballot_leader_election::Ballot;
use omnipaxos::leader_election::ballot_leader_election::BallotLeaderElection;
use omnipaxos::messages::AcceptDecide;
use omnipaxos::messages::AcceptSync;
use omnipaxos::messages::Accepted;
use omnipaxos::messages::Decide;
use omnipaxos::messages::FirstAccept;
use omnipaxos::messages::Message;
use omnipaxos::messages::PaxosMsg;
use omnipaxos::messages::Prepare;
use omnipaxos::messages::Promise;
use omnipaxos::paxos::ProposeErr;
use omnipaxos::storage::memory_storage::MemorySequence;
use omnipaxos::storage::memory_storage::MemoryState;
use omnipaxos::utils::hocon_kv::CONFIG_ID;
use omnipaxos::utils::hocon_kv::HB_DELAY;
use omnipaxos::utils::hocon_kv::INITIAL_DELAY_FACTOR;
use omnipaxos::utils::hocon_kv::LOG_FILE_PATH;
use omnipaxos::utils::hocon_kv::PID;
use omnipaxos::utils::hocon_kv::PRIORITY;
use omnipaxos::utils::logger::create_logger;
type OmniPaxos<T> = omnipaxos::paxos::OmniPaxos<T, MemorySequence<T>, MemoryState>;
use std::pin::Pin

use arc_runtime::prelude::*;

use std::collections::HashMap;
use std::time::Duration;

use crate::ble::BallotLeaderElectionComp;
use crate::omni::OmniPaxosComp;

const START_TIMEOUT: Duration = Duration::from_millis(1000);
const REGISTRATION_TIMEOUT: Duration = Duration::from_millis(1000);
const STOP_COMPONENT_TIMEOUT: Duration = Duration::from_millis(1000);
const BLE_TIMER_TIMEOUT: Duration = Duration::from_millis(100);

pub struct System<T: Data> {
    pub(crate) kompact: KompactSystem,
    pub(crate) nodes: HashMap<u64, Node<T>>,
}

#[derive(New)]
pub struct Node<T: Data> {
    pub ble: Arc<Component<BallotLeaderElectionComp>>,
    pub omni: Arc<Component<OmniPaxosComp<T>>>,
}

fn build_kompact_system(num_threads: usize) -> KompactSystem {
    let mut conf = KompactConfig::default();
    conf.set_config_value(&THREADS, num_threads);

    match num_threads {
        _ if num_threads <= 32 => conf.executor(|t| crossbeam_workstealing_pool::small_pool(t)),
        _ if num_threads <= 64 => conf.executor(|t| crossbeam_workstealing_pool::large_pool(t)),
        _ => conf.executor(|t| crossbeam_workstealing_pool::dyn_pool(t)),
    };

    let mut net = NetworkConfig::default();
    net.set_tcp_nodelay(true);

    conf.system_components(DeadletterBox::new, net.build());
    conf.build().unwrap()
}

impl<T: Data> System<T> {
    pub fn new(
        num_nodes: usize,
        num_threads: usize,
        ble_hb_delay: u64,
        ble_initial_delay_factor: Option<u64>,
        ble_initial_leader: Option<Ballot>,
    ) -> Self {
        let kompact = build_kompact_system(num_threads);

        let mut nodes = HashMap::new();
        let mut pids = (1..=num_nodes as u64).collect::<Vec<_>>();
        let mut ble_refs = HashMap::new();
        let mut omni_refs = HashMap::new();

        for pid in 1..=num_nodes as u64 {
            let mut peer_pids = pids.clone();
            peer_pids.retain(|peer| peer != &pid);

            let priority = None;
            let logger = None;
            let log_file_path = None;

            let (ble_comp, ble_path) = kompact.create_and_register(|| {
                BallotLeaderElectionComp::new(BallotLeaderElection::with(
                    pid,
                    peer_pids.clone(),
                    priority,
                    ble_hb_delay,
                    ble_initial_leader,
                    ble_initial_delay_factor,
                    logger,
                    log_file_path,
                ))
            });

            let config_id = 1;
            let skip_prepare_use_leader = None;
            let logger = None;
            let log_file_path = None;

            let (omni_comp, omni_path) = kompact.create_and_register(|| {
                OmniPaxosComp::new(OmniPaxos::with(
                    config_id,
                    pid,
                    peer_pids.clone(),
                    skip_prepare_use_leader,
                    logger,
                    log_file_path,
                ))
            });

            biconnect_components(&ble_comp, &omni_comp);

            ble_path.wait_timeout(REGISTRATION_TIMEOUT).unwrap();
            omni_path.wait_timeout(REGISTRATION_TIMEOUT).unwrap();

            ble_refs.insert(pid, ble_comp.actor_ref());
            omni_refs.insert(pid, omni_comp.actor_ref());

            nodes.insert(pid, Node::new(ble_comp, omni_comp));
        }

        for node in nodes.values() {
            node.ble.on_definition(|c| c.peers = ble_refs.clone());
            node.omni.on_definition(|c| c.peers = omni_refs.clone());
        }

        Self { kompact, nodes }
    }

    pub fn start_all_nodes(&self) {
        for node in self.nodes.values() {
            self.kompact
                .stop_notify(&node.ble)
                .wait_timeout(STOP_COMPONENT_TIMEOUT)
                .unwrap();

            self.kompact
                .start_notify(&node.omni)
                .wait_timeout(START_TIMEOUT)
                .unwrap();
        }
    }

    pub fn kill_node(&mut self, id: u64) {
        let node = self.nodes.remove(&id).unwrap();
        self.kompact
            .kill_notify(node.ble)
            .wait_timeout(STOP_COMPONENT_TIMEOUT)
            .unwrap();

        self.kompact
            .kill_notify(node.omni)
            .wait_timeout(STOP_COMPONENT_TIMEOUT)
            .unwrap();
    }
}
