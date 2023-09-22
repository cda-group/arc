#![allow(unused)]
use runtime::prelude::*;

fn main() {
    let db = Database::new("snapshots");
    Runner::new(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/log")).spawn(instance(db.clone()));
}

async fn instance(_db: Database) {
    let s0 = Stream::<Yellow>::source(
        Reader::file(
            Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/data/csv/yellow/yellow_tripdata_2017-01.csv"
            )),
            false,
        ),
        Encoding::csv(','),
        TimeSource::event(
            |r| r.tpep_pickup_datetime,
            Duration::from_seconds(10), // Watermark interval
            Duration::from_seconds(2),  // Slack
        ),
    );

    let s1 = s0.filter(|r: Yellow| {
        {
            r.passenger_count > 0
                && r.passenger_count > 2
                && r.trip_distance > 0.1
                && r.trip_distance < 22.58
                && r.pu_location_id > 0
                && r.do_location_id > 0
                && r.payment_type > 0
                && r.fare_amount > 0.0
                && r.tpep_pickup_datetime.year() >= 2017
                && r.tpep_pickup_datetime.year() < 2018
        }
        .into()
    });

    let s2 = s1.map(|r: Yellow| R2 {
        passenger_count: r.passenger_count,
        pu_location_id: r.pu_location_id,
        do_location_id: r.do_location_id,
        trip_distance: r.trip_distance,
        fare_amount: r.fare_amount,
        payment_type: r.payment_type,
    });

    let locations: Vec<Location> = File::open(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "taxi+_zone_lookup.csv"
    )))
    .read_to_string()
    .decode(Encoding::csv(','));

    let s3 = s2.keyby(|r: R2| r.do_location_id);

    s3.unkey().sink(
        Writer::file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/output.csv"
        ))),
        Encoding::csv(','),
    );
}

use runtime::*;

#[data]
pub struct Green {
    /// A code indicating the TPEP provider that provided the record.
    /// 1 => Creative Mobile Technologies, LLC;
    /// 2 => VeriFone Inc.
    pub vendor_id: u8,
    /// The date and time when the meter was engaged.
    pub lpep_pickup_datetime: Time,
    /// The date and time when the meter was disengaged.
    pub lpep_dropoff_datetime: Time,
    /// This flag indicates whether the trip record was held in vehicle
    /// memory before sending to the vendor, aka “store and forward,”
    /// because the vehicle did not have a connection to the server.
    /// Y=store and forward trip
    /// N=not a store and forward trip
    pub store_and_fwd_flag: Flag,
    /// The final rate code in effect at the end of the trip.
    /// 1 => Standard rate
    /// 2 => JFK
    /// 3 => Newark
    /// 4 => Nassau or Westchester
    /// 5 => Negotiated fare
    /// 6 => Group ride
    pub ratecode_id: u8,
    /// TLC Taxi Zone in which the taximeter was engaged.
    pub pu_location_id: u64,
    /// TLC Taxi Zone in which the taximeter was disengaged.
    pub do_location_id: u64,
    /// The number of passengers in the vehicle.
    pub passenger_count: u64,
    /// The elapsed trip distance in miles reported by the taximeter.
    pub trip_distance: f64,
    /// The time-and-distance fare calculated by the meter.
    pub fare_amount: f64,
    /// Miscellaneous extras and surcharges. Currently, this only includes
    /// the $0.50 and $1 rush hour and overnight charges.
    pub extra: f64,
    /// $0.50 MTA tax that is automatically triggered based on the metered
    /// rate in use.
    pub mta_tax: f64,
    /// Tip amount – This field is automatically populated for credit card
    /// tips. Cash tips are not included.
    pub tip_amount: f64,
    /// Total amount of all tolls paid in trip.
    pub tolls_amount: f64,
    pub ehail_fee: f64,
    /// $0.30 improvement surcharge assessed trips at the flag drop. The
    /// improvement surcharge began being levied in 2015.
    pub improvement_surcharge: f64,
    /// The total amount charged to passengers. Does not include cash tips.
    pub total_amount: f64,
    /// A numeric code signifying how the passenger paid for the trip.
    /// 1 => Credit card
    /// 2 => Cash
    /// 3 => No charge
    /// 4 => Dispute
    /// 5 => Unknown
    /// 6 => Voided trip
    pub payment_type: u8,
    pub trip_type: u8,
    pub congestion_surcharge: f64,
}

#[data]
pub struct Yellow {
    /// A code indicating the TPEP provider that provided the record.
    /// 1 => Creative Mobile Technologies, LLC;
    /// 2 => VeriFone Inc.
    pub vendor_id: u8,
    /// The date and time when the meter was engaged.
    pub tpep_pickup_datetime: Time,
    /// The date and time when the meter was disengaged.
    pub tpep_dropoff_datetime: Time,
    /// The number of passengers in the vehicle.
    pub passenger_count: u64,
    /// The elapsed trip distance in miles reported by the taximeter.
    pub trip_distance: f64,
    /// The final rate code in effect at the end of the trip.
    /// 1 => Standard rate
    /// 2 => JFK
    /// 3 => Newark
    /// 4 => Nassau or Westchester
    /// 5 => Negotiated fare
    /// 6 => Group ride
    pub ratecode_id: u8,
    /// This flag indicates whether the trip record was held in vehicle
    /// memory before sending to the vendor, aka “store and forward,”
    /// because the vehicle did not have a connection to the server.
    /// Y=store and forward trip
    /// N=not a store and forward trip
    pub store_and_fwd_flag: Flag,
    /// TLC Taxi Zone in which the taximeter was engaged.
    pub pu_location_id: u64,
    /// TLC Taxi Zone in which the taximeter was disengaged.
    pub do_location_id: u64,
    /// A numeric code signifying how the passenger paid for the trip.
    /// 1 => Credit card
    /// 2 => Cash
    /// 3 => No charge
    /// 4 => Dispute
    /// 5 => Unknown
    /// 6 => Voided trip
    pub payment_type: u8,
    /// The time-and-distance fare calculated by the meter.
    pub fare_amount: f64,
    /// Miscellaneous extras and surcharges. Currently, this only includes
    /// the $0.50 and $1 rush hour and overnight charges.
    pub extra: f64,
    /// $0.50 MTA tax that is automatically triggered based on the metered
    /// rate in use.
    pub mta_tax: f64,
    /// Tip amount – This field is automatically populated for credit card
    /// tips. Cash tips are not included.
    pub tip_amount: f64,
    /// Total amount of all tolls paid in trip.
    pub tolls_amount: f64,
    /// $0.30 improvement surcharge assessed trips at the flag drop. The
    /// improvement surcharge began being levied in 2015.
    pub improvement_surcharge: f64,
    /// The total amount charged to passengers. Does not include cash tips.
    pub total_amount: f64,
}

#[data]
pub struct R2 {
    pub passenger_count: u64,
    pub trip_distance: f64,
    pub pu_location_id: u64,
    pub do_location_id: u64,
    pub payment_type: u8,
    pub fare_amount: f64,
}

#[data]
pub struct R3 {
    pub total: f64,
}

// 1. Filter out rides which have negative distance
// 2. Map to Record2
// 3. Aggregate the total number of trips per hour per pickup location.
// 4. Group by number of trips

#[data]
pub struct Location {
    pub location_id: u64,
    pub borough: String,
    pub zone: String,
    pub service_zone: String,
}

#[data]
pub struct R5 {
    #[serde(flatten)]
    pub r3: R3,
    #[serde(flatten)]
    pub r4: Location,
}

#[data]
pub enum Flag {
    Y,
    N,
}
