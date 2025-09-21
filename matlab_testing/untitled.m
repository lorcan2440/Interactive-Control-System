% Parameters
k12 = 10;
k21 = 20;
d   = 1;

% PID controller
Kp = 6.3;
Ki = 350;
Kd = 0;

% State-space matrices
A = [-(k12+d),  k21;
      k12   , -(k21+d)];
B = [1; 0];
C = [0 1];
D = 0;

% Calculate plant TF
sys_ss = ss(A, B, C, D);
G = tf(sys_ss);

% Calculate controller TF
K = tf([Kd, Kp, Ki], [1, 0]);

% Calculate OLTF
L = K * G;

% Closed-loop TF with unity negative feedback
CL = feedback(L, 1);

% Characteristic polynomial and roots
char_poly = cell2mat(CL.Denominator); % denominator coefficients
disp('Characteristic polynomial coefficients:')
disp(char_poly)

disp('Closed-loop poles (roots of 1+L=0):')
disp(roots(char_poly))