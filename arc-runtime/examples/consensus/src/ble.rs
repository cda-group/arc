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

use arc_runtime::prelude::*;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Duration;

use crate::port::BallotPort;

const START_TIMEOUT: Duration = Duration::from_millis(1000);
const REGISTRATION_TIMEOUT: Duration = Duration::from_millis(1000);
const STOP_COMPONENT_TIMEOUT: Duration = Duration::from_millis(1000);
const BLE_TIMER_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(ComponentDefinition, Deref, DerefMut)]
pub struct BallotLeaderElectionComp {
    ctx: ComponentContext<Self>,
    ble_port: ProvidedPort<BallotPort>,
    pub(crate) peers: HashMap<u64, ActorRef<BLEMessage>>,
    pub leader: Option<Ballot>,
    timer: Option<ScheduledTimer>,
    pub(crate) asks: VecDeque<Ask<(), Ballot>>,
    #[deref]
    #[deref_mut]
    ble: BallotLeaderElection,
}

impl BallotLeaderElectionComp {
    pub fn new(ble: BallotLeaderElection) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            ble_port: ProvidedPort::uninitialised(),
            peers: HashMap::new(),
            leader: None,
            timer: None,
            ble,
            asks: VecDeque::new(),
        }
    }

    fn send_outgoing_msgs(&mut self) {
        for msg in self.ble.get_outgoing_msgs() {
            self.peers.get(&msg.to).unwrap().tell(msg)
        }
    }

    fn answer_future(&mut self, b: Ballot) {
        if let Some(ask) = self.asks.pop_front() {
            ask.reply(b).unwrap();
        }
    }
}

impl Actor for BallotLeaderElectionComp {
    type Message = BLEMessage;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        self.ble.handle(msg);
        Handled::Ok
    }

    fn receive_network(&mut self, msg: NetMessage) -> Handled {
        todo!()
    }
}

impl ComponentLifecycle for BallotLeaderElectionComp {
    fn on_start(&mut self) -> Handled {
        self.ble.new_hb_round();
        self.timer = self
            .schedule_periodic(BLE_TIMER_TIMEOUT, BLE_TIMER_TIMEOUT, move |c, _| {
                if let Some(l) = c.ble.tick() {
                    c.answer_future(l);
                    c.ble_port.trigger(l);
                }
                c.send_outgoing_msgs();
                Handled::Ok
            })
            .into();
        Handled::Ok
    }

    fn on_stop(&mut self) -> Handled {
        if let Some(timer) = self.timer.take() {
            self.cancel_timer(timer);
        }
        Handled::Ok
    }

    fn on_kill(&mut self) -> Handled {
        Handled::Ok
    }
}

impl Provide<BallotPort> for BallotLeaderElectionComp {
    fn handle(&mut self, _: Never) -> Handled {
        // ignore
        Handled::Ok
    }
}
