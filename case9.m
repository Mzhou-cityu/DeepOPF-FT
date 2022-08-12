% In the modified IEEE 9-bus system, we assume the lines between bus 3-6, 2-8, 1-4 
% are perfect lines. Then, we merge the buses 3-6, 2-8, 1-4 together and renumber 
% the buses. In this way, we can evaluate the perforance of DeepOPF-FT over all possible
% topologies and with the same bus, generation, and line capacity configurations.
function mpc = case9
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
    1	3	0.0000 	0.0000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000;
    2	2	0.0000 	0.0000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000; 
    3	2	0.0000 	0.0000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000; 
    4	1	90.0000 	45.0000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000; 
    5	1	100.0000 	52.5000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000;
    6	1	125.0000 	75.0000 	0.0000 	0.000 	1 	1.0000 	0.0000 	345	1	1.1000 	0.9000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
    1       72.3	27.03	210     -210	1.040 	100.0 	1	250     10;
    2       163     6.54	210     -210	1.025 	100.0 	1	300     10;
    3       85      -10.95	210     -210	1.025 	100.0 	1	270     10;
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
    2	1500	0	3	0.1100 	5.000 	950.0; 
    2	2000	0	3	0.0850 	1.200 	1400.0; 
    2	3000	0	3	0.1225 	1.000 	1135.0; 
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
    1	2	0.0000 	0.0576 	0.0000 	250.000 	250.0 	250	0	0	1	-360	360;
    1	3	0.0000 	0.0576 	0.0000 	250.000 	250.0 	250	0	0	1	-360	360;
    1	4	0.0000 	0.0576 	0.0000 	250.000 	250.0 	250	0	0	1	-360	360;
    1	5	0.0000 	0.0586 	0.0000 	300.000 	300.0 	300	0	0	1	-360	360;
    1	6	0.0170 	0.0920 	0.1580 	250.000 	250.0 	250	0	0	1	-360	360;
    2	3	0.0570 	0.0920 	0.1580 	250.000 	250.0 	250	0	0	1	-360	360;
    2	4	0.0570 	0.0920 	0.1580 	250.000 	250.0 	250	0	0	1	-360	360;
    2	5	0.1390 	0.1700 	0.3580 	150.000 	150.0 	150	0	0	1	-360	360;
    2	6	0.1390 	0.1700 	0.3580 	150.000 	150.0 	150	0	0	1	-360	360;
    3	4	0.0100 	0.0850 	0.1760 	250.000 	250.0 	250	0	0	1	-360	360;
    3	5	0.0560 	0.0586 	0.0000 	300.000 	300.0 	300	0	0	1	-360	360;
    3	6	0.1760 	0.2586 	0.0000 	300.000 	300.0 	300	0	0	1	-360	360;
    4	5	0.1560 	0.2586 	0.0000 	300.000 	300.0 	300	0	0	1	-360	360;
    4	6	0.1856 	0.2586 	0.0000 	300.000 	300.0 	300	0	0	1	-360	360;
    5	6	0.0119 	0.1008 	0.2090 	150.000 	150.0 	150	0	0	1	-360	360;
];
