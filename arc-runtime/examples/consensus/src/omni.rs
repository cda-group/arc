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
use omnipaxos::storage::Entry;
use omnipaxos::storage::PaxosState;
use omnipaxos::storage::Sequence;
use omnipaxos::storage::StopSign;
use omnipaxos::utils::hocon_kv::CONFIG_ID;
use omnipaxos::utils::hocon_kv::HB_DELAY;
use omnipaxos::utils::hocon_kv::INITIAL_DELAY_FACTOR;
use omnipaxos::utils::hocon_kv::LOG_FILE_PATH;
use omnipaxos::utils::hocon_kv::PID;
use omnipaxos::utils::hocon_kv::PRIORITY;
use omnipaxos::utils::logger::create_logger;
type OmniPaxos<T> = omnipaxos::paxos::OmniPaxos<T, MemorySequence<T>, MemoryState>;

use arc_runtime::prelude::*;

use crate::port::BallotPort;

use std::collections::HashMap;
use std::collections::VecDeque;

use std::time::Duration;

use std::ops::Deref;
use std::ops::DerefMut;

const OMNI_TIMER_TIMEOUT: Duration = Duration::from_millis(1);

#[derive(ComponentDefinition, Deref, DerefMut)]
pub struct OmniPaxosComp<T: Data> {
    ctx: ComponentContext<Self>,
    ble_port: RequiredPort<BallotPort>,
    pub(crate) peers: HashMap<u64, ActorRef<Message<T>>>,
    timer: Option<ScheduledTimer>,
    pub(crate) asks: VecDeque<Ask<(), Entry<T>>>,
    #[deref]
    #[deref_mut]
    paxos: OmniPaxos<T>,
}

impl<T: Data> OmniPaxosComp<T> {
    pub fn new(paxos: OmniPaxos<T>) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            ble_port: RequiredPort::uninitialised(),
            peers: HashMap::new(),
            timer: None,
            paxos,
            asks: VecDeque::new(),
        }
    }

    pub fn send_outgoing_msgs(&mut self) {
        for msg in self.paxos.get_outgoing_msgs() {
            self.peers.get(&msg.to).unwrap().tell(msg);
        }
    }

    fn answer_future(&mut self) {
        if !self.asks.is_empty() {
            for entry in self.paxos.get_latest_decided_entries() {
                self.asks.pop_front().unwrap().reply(entry.clone());
            }
        }
    }
}

impl<T: Data> Actor for OmniPaxosComp<T> {
    type Message = Message<T>;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        self.paxos.handle(msg);
        Handled::Ok
    }

    fn receive_network(&mut self, msg: NetMessage) -> Handled {
        todo!()
    }
}

impl<T: Data> ComponentLifecycle for OmniPaxosComp<T> {
    fn on_start(&mut self) -> Handled {
        self.timer = self
            .schedule_periodic(OMNI_TIMER_TIMEOUT, OMNI_TIMER_TIMEOUT, move |c, _| {
                c.send_outgoing_msgs();
                c.answer_future();
                Handled::Ok
            })
            .into();
        Handled::Ok
    }

    fn on_kill(&mut self) -> Handled {
        if let Some(timer) = self.timer.take() {
            self.cancel_timer(timer);
        }
        Handled::Ok
    }
}

impl<T: Data> Require<BallotPort> for OmniPaxosComp<T> {
    fn handle(&mut self, msg: Ballot) -> Handled {
        self.paxos.handle_leader(msg);
        Handled::Ok
    }
}
