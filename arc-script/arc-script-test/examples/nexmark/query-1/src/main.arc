-- Types (Entities)

type Person        = { name, email_address, credit_card, city, state }
type Item          = { description, name }
type OpenAuction   = { initial, reserve, expires }
type ClosedAuction = { date }
type Bid           = { item_id, bidder_id, time, price, date }
type Category      = { name, description }

-- Tasks (Business Objects)

task Map(f) (In) -> (Out) {
    on In(x) => emit Out(f(x))
}

task Auction() () -> () {
    
}

task Bid() () -> () {

}

extern fn dol_to_eur(dollar);

fn query1(bids: Stream[Bid]) {
    let people: Stream[Person] = source::People() () in
    let items: Stream[Item] = source::Items() () in
    let bids: Stream[Bid] = source::Bids() () in

    let mapped = Map(|bid| {
        item_id = bid.item_id,
        bidder_id = bid.bidder_id,
        eur = dol_to_eur(bid.price),
        time = bid.time
    }) (items) in

    sink::Query1() (mapped)
}
