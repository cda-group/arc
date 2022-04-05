# [Pulse Detection](https://dl.acm.org/doi/pdf/10.1145/3428251) (Not yet fully supported)

> Arterial Blood Pressure (ABP) pulse detection [O’Rourke 1971; Zong et al. 2003] is a complex streaming computation, and is difficult to express with existing languages for stream processing. The use of a streaming query language for medical monitoring applications has been considered in [Abbas et al. 2018, 2019].

> The ABP signal is collected from the MIT-BIH Polysomnographic database [Ichimaru and Moody 1999]. The signal measurements are of `type VT = {val: V, ts: T}`, where `val` is the value of the signal and `ts` is the timestamp. The signal is uniformly sampled at a frequency of `250Hz`. (...) The ABP waveform contains rich information about the cardiovascular system (e.g., heart rate, systolic, mean, and diastolic arterial pressures). Reliable ABP pulse detection is crucial for extracting this information. 
> First, the algorithm preprocesses the signal stream using a low-pass **IIR** filter and a slope sum function (**SSF**), and then it performs the detection of the pulse onset.

> The low-pass filter **IIR** suppresses high frequency noise, and is defined by `𝑦(𝑛)=2𝑦(𝑛−1)−𝑦(𝑛−2)+𝑥(𝑛)−2𝑥(𝑛−5)+𝑥(𝑛−10)`. The **SSF** is defined by `𝑧(𝑛)=Σ0≤𝑖≤31𝑚𝑎𝑥(0,𝑑(𝑛−𝑖))`, where `𝑑(𝑛)=𝑦(𝑛)−𝑦(𝑛−1)`. It enhances the up-slope of the ABP pulse and restrains the remainder of the pressure waveform. The query `getVTP : Q(VT, VTP)` annotates each item `{val, ts}` of the input stream with an additional component `pval`, which is the result of the preprocessing. The `type VTP = {val: V, ts: T, pval: V}` extends `VT` with this additional component. These preprocessed values have a phase shift of `20ms` (`5` samples), which is introduced by low-pass filtering.

> The detection of ABP onset is described by the following rules:
> * **R1**. In intervals where the SSF value exceeds a threshold `Thred` (i.e. a tentative pulse), the algorithm selects the *first* and the *maximum* SSF values.
> * **R2**. The pulse detection is accepted only if the *difference* between the first and the maximum SSF values exceeds `100`.
> * **R3**. When the pulse is accepted, the algorithm chooses the first sample that crosses the threshold as the onset point. The detected onset is adjusted by `20ms` (`5` samples) to compensate for the phase shift of low-pass filtering.
> * **R4**. After an onset is detected, to avoid double detection of the same pulse, the detection falls silent for `300ms`.


## Implementation (arc-lang)

```arc-lang
type V = i32;
type T = time;
type VT = {val: V, ts: T};
type VTP = {val: V, ts: T, pval: V};

fun query(stream: ~VT, Thred: V) -> ~PTV {
    val lowPass = IIR({−1,2}, {1,0,0,0,0,−2,0,0,0,0,1});
    val diff = SWindow(2, 1, fun(x): x[1] - x[0]);
    val sum = SWindow(32, 1, Reduce(fun(x, y): if y > 0 { x + y } else { x }))
    val ssf = diff |> sum;
    val preProc = Map(fun(x): x.val) |> lowPass |> ssf;
    val getVTP = Zip(fun(x, y): {x.val, x.ts, y}) (stream, preProc);

    # R1
    val pulse = TakeWhen(fun(x): x.pval > Thred, fun(x): x.pval < Thred);
    val select = Reduce(fun(x): (x, x), ())

    # TODO

}
```

## Implementation ([StreamQL](https://dl.acm.org/doi/pdf/10.1145/3428251))

```text
# Pre-process the signal
lowPass = IIR({−1,2}, {1,0,0,0,0,−2,0,0,0,0,1})

diff = sWindow(2, 1, (x, y) -> y − x)

sum = sWindow(32, 1, reduce((x, y) -> (y > 0) ? (x + y) : x))

ssf = diff ≫ sum

preProc = map(x -> x.val) ≫ lowPass ≫ ssf

getVTP = annotate(preProc, (x, y) -> ⟨x.val, x.ts, y⟩)

# select signal interval containing a peak (R1)
pulse = takeWhen(x -> x.pval > Thred, x -> x.pval < Thred)

# select the first element in interval as the onset sample
# find the measurement with the maximum preprocessed value,
# and store them as a pair ⟨first, max⟩
select = reduce(x -> ⟨x, x⟩,
                     ( ⟨f, m⟩, x) -> ⟨f, (x.pval > m.pval) ? x : m⟩)

# examine the detected pulse (R2) and project the onset
getOnset = filterMap( ⟨f, m⟩ -> m.pval − f.pval > 100, ⟨f, m⟩ -> f)

detect1 = getVTP ≫ pulse ≫ select ≫ check ≫ getOnset

rft = skip(75)  # after detecting the ABP onset, apply R4

detectAll = seq(detect1, iter(rft ≫ detect1))

subShift = map(x -> x.ts − 5)  # compensate for phase shift

ABPDetection = detectAll ≫ subShift
```
