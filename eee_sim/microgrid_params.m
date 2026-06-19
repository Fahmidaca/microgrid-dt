%% ========================================================================
%  microgrid_params.m
%  ------------------------------------------------------------------------
%  Defines the parameter struct P and the weather input timeseries
%  (G_profile, T_profile) that build_microgrid_pq_digital_twin.m needs.
%
%  Bangladesh context: 50 Hz / 230 V grid, Dhaka-like clear-sky solar.
%  Sample step Ts = 1e-4 s (10 kHz) - enough for 5th..25th harmonic FFT
%  on a 50 Hz fundamental and clean integer-cycle window math.
% =========================================================================

P = struct();

% ---- simulation clock --------------------------------------------------
P.Ts        = 1e-4;          % fixed solver step  [s]   (10 kHz)
P.fs        = 1/P.Ts;        % sample rate        [Hz]
P.f0        = 50;            % grid fundamental   [Hz]  (Bangladesh)
P.Vnom      = 230;           % nominal phase RMS  [V]
P.Tend      = 1.00;          % total sim time     [s]

% ---- voltage sag event -------------------------------------------------
P.sag_start = 0.30;          % sag begins         [s]
P.sag_dur   = 0.10;          % sag duration       [s]
P.sag_depth = 0.30;          % 30 %% sag (V -> 70 %% of nominal)

% ---- APF activation step -----------------------------------------------
P.apf_on_t  = 0.50;          % APF turns ON at    [s]

% ---- nonlinear load (6-pulse rectifier) --------------------------------
P.Ifund     = 15;            % fundamental load current   [A]
P.harm_ord  = [5  7  11 13]; % characteristic harmonic orders
P.harm_amp  = [0.20 0.14 0.09 0.077];  % relative magnitudes

% ---- PV array (10 kWp) -------------------------------------------------
P.Pstc      = 10000;         % rated power at STC         [W]
P.gam_T     = -0.0040;       % power temp coefficient    [1/K]
P.NOCT      = 45;            % nominal operating cell T  [degC]

% ---- battery + economics (Bangladesh tariff) ---------------------------
P.Batt_kWh  = 20;            % battery capacity           [kWh]
P.cost_kWh  = 24000;         % replacement cost           [BDT/kWh]
P.base_fade = 8e-7;          % SoH fraction lost per kWh throughput

% =========================================================================
%  Weather input timeseries - From-Workspace blocks expect [t value] rows.
% =========================================================================
N = ceil(P.Tend/P.Ts) + 1;
t = (0:N-1).' * P.Ts;

% Solar irradiance: 800 -> 950 W/m^2 ramp over the run (passing cloud edge)
G_vec = linspace(800, 950, N).';
% Ambient temperature: 32 -> 34 degC slow drift
T_vec = linspace(32, 34, N).';

G_profile = [t, G_vec];
T_profile = [t, T_vec];

% =========================================================================
%  Push everything to base workspace so the build script can pick it up.
% =========================================================================
assignin('base','P',           P);
assignin('base','G_profile',   G_profile);
assignin('base','T_profile',   T_profile);

fprintf('[microgrid_params] P, G_profile (%dx2), T_profile (%dx2) loaded.\n', N, N);
fprintf('[microgrid_params] f0=%d Hz, Vnom=%d V, Ts=%g s, Tend=%g s.\n', ...
        P.f0, P.Vnom, P.Ts, P.Tend);
