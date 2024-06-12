# [Under Review] ExtNCO: A fine-grained divide-and-conquer approach for extending NCO to solve large-scale Traveling Salesman Problem

![overview](./overview.png)

This repository contains the implementation of our paper ExtNCO: A fine-grained divide-and-conquer approach for extending NCO to solve large-scale Traveling Salesman Problem.

## Highlights

* A divide-and-conquer approach is proposed to extend NCO to solve **large-scale TSP**.
* An **eﬀective dividing strategy** is proposed to construct small-scall sub-problems.
* A **MST-based merging strategy** is proposed for better solution quality.
* POMO trained on TSP-100 is employed to **solve TSP-1M**, achieveing **optimal gap < 6%**.

## Dependencies

#### Main Dependency Packages

* [python](https://www.python.org) >= 3.8

* [pytorch](https://pytorch.org) >= 2.0.1

* [cuda](https://developer.nvidia.com/cuda-toolkit) >= 12.2

* [LKH](http://webhotel4.ruc.dk/~keld/research/LKH-3/) >= 3.0.7

* [tsplib95](https://github.com/rhgrant10/tsplib95)

* [pylkh](https://github.com/ben-hudson/pylkh)

* [mpire](https://github.com/sybrenjansen/mpire)

* [numpy](https://numpy.org)

* [tqdm](https://github.com/tqdm/tqdm#table-of-contents)

additional packages may be required to reproduce baselines

## Basic Usage

We provide codes of ExtNCO that utilizes POMO pre-trained on TSP-100 to solve REI / TSPLIB / VLSI instances.

```python
python main.py
```
To run the comparative study, you need to specify the following parameters in `main.py`:
* `{method}`: 'extnco-eff'/ 'extnco-bal'/ 'extnco-qlt'/ 'htsp'/ 'pomo'/ 'deepaco'/ 'difusco'/ 'difusco-2opt'/ 'lkh'
* `{stage}`: 'first'/ 'second'
* `{dataset}`: 'rei'/ 'tsplib'/ 'vlsi'

## LKH
We provide source codes of [LKH-3.0.7](http://webhotel4.ruc.dk/~keld/research/LKH-3/), and you need to install it first.
For example, on Ubuntu:
```bash
cd LKH-3.0.7/
make
```
