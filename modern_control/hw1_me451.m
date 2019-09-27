close all; clear;

A = -0.03;
B = [1.2 -9.81];
C = 1.0;
D = [0 0];

Kp = [0 1];
ctrl = pid(Kp);

sys = ss(A,B,C,D);
% sysl = tf(sys);

closedLoop = feedback(sys*ctrl,1);

step(closedLoop, 'g--');