%% ========================================================================
%  build_microgrid_pq_digital_twin.m  (cleaned)
%  ------------------------------------------------------------------------
%  Programmatically builds a runnable Simulink model of a Bangladesh
%  renewable microgrid power-quality digital twin.
%
%  Changes vs. original draft:
%    * Hard-coded Ts removed from APF / PQ / Twin codes - Ts is now
%      injected from P via a base-workspace masked constant.
%    * FFT bin formula computed from Ts/N rather than fixed 5*k+1.
%    * P passed in as argument instead of evalin().
%    * arrangeSystem call wrapped in try/catch for backwards compat.
%
%  REQUIREMENTS: MATLAB R2021b+ with Simulink.
%
%  USAGE:
%    >> microgrid_params
%    >> build_microgrid_pq_digital_twin
%    >> sim('microgrid_pq_twin')
% =========================================================================
function build_microgrid_pq_digital_twin(P)

if nargin < 1
    if ~evalin('base','exist(''P'',''var'')')
        evalin('base','microgrid_params');
    end
    P = evalin('base','P');
end

mdl = 'microgrid_pq_twin';

% start clean
if any(strcmp(find_system('SearchDepth',0),mdl)), close_system(mdl,0); end
new_system(mdl);
open_system(mdl);

% fixed-step discrete solver (required for FFT/SRF persistent states)
set_param(mdl, ...
    'SolverType','Fixed-step','Solver','FixedStepDiscrete', ...
    'FixedStep',num2str(P.Ts),'StopTime',num2str(P.Tend));

% =========================================================================
%  1. SOURCES
% =========================================================================
addb([mdl '/Clock'],'simulink/Sources/Clock',[30 200 60 230]);

addb([mdl '/Irradiance G'],'simulink/Sources/From Workspace',[30 60 110 90]);
set_param([mdl '/Irradiance G'],'VariableName','G_profile');

addb([mdl '/Ambient T'],'simulink/Sources/From Workspace',[30 110 110 140]);
set_param([mdl '/Ambient T'],'VariableName','T_profile');

addb([mdl '/Voltage Sag'],'simulink/Sources/Pulse Generator',[30 300 110 340]);
set_param([mdl '/Voltage Sag'], ...
    'PulseType','Time based','Amplitude','1', ...
    'Period',num2str(P.Tend), ...
    'PulseWidth',num2str(100*P.sag_dur/P.Tend), ...
    'PhaseDelay',num2str(P.sag_start));

addb([mdl '/APF Enable'],'simulink/Sources/Step',[300 360 360 390]);
set_param([mdl '/APF Enable'], ...
    'Time',num2str(P.apf_on_t),'Before','0','After','1');

% Constant Ts source - propagated everywhere instead of hard-coding it.
addb([mdl '/Ts const'],'simulink/Sources/Constant',[30 380 110 410]);
set_param([mdl '/Ts const'],'Value',num2str(P.Ts));

% =========================================================================
%  2. MATLAB FUNCTION BLOCKS
% =========================================================================
addMLfun(mdl,'Microgrid Plant',  plantCode(),  [180 80  300 180]);
addMLfun(mdl,'SRF Active Filter',apfCode(),    [400 120 520 220]);
addMLfun(mdl,'FFT PQ Analyzer',  pqCode(),     [600 90  730 220]);
addMLfun(mdl,'Digital Twin',     twinCode(),   [800 90  930 220]);

% =========================================================================
%  3. SINKS
% =========================================================================
addb([mdl '/THD Scope'],'simulink/Sinks/Scope', [1000 60 1040 100]);
addb([mdl '/Econ Scope'],'simulink/Sinks/Scope',[1000 150 1040 190]);
addb([mdl '/Mux THD'],'simulink/Signal Routing/Mux',[940 60 945 100]);
set_param([mdl '/Mux THD'],'Inputs','2');
addb([mdl '/Mux Econ'],'simulink/Signal Routing/Mux',[940 150 945 190]);
set_param([mdl '/Mux Econ'],'Inputs','2');

addb([mdl '/log_THD_i'],'simulink/Sinks/To Workspace',[1000 240 1080 270]);
set_param([mdl '/log_THD_i'], ...
    'VariableName','THD_i_log','SaveFormat','Timeseries');
addb([mdl '/log_cost'],'simulink/Sinks/To Workspace',[1000 290 1080 320]);
set_param([mdl '/log_cost'], ...
    'VariableName','cost_log','SaveFormat','Timeseries');

% =========================================================================
%  4. WIRING
% =========================================================================
ph = @(b) get_param([mdl '/' b],'PortHandles');

clk  = ph('Clock');
gW   = ph('Irradiance G');
tW   = ph('Ambient T');
sag  = ph('Voltage Sag');
apfE = ph('APF Enable');
tsC  = ph('Ts const');
plt  = ph('Microgrid Plant');
apf  = ph('SRF Active Filter');
pq   = ph('FFT PQ Analyzer');
tw   = ph('Digital Twin');
mTHD = ph('Mux THD');
mEco = ph('Mux Econ');

% Plant: (t, G, T, sag)
connect(mdl, clk.Outport(1), plt.Inport(1));
connect(mdl, gW.Outport(1),  plt.Inport(2));
connect(mdl, tW.Outport(1),  plt.Inport(3));
connect(mdl, sag.Outport(1), plt.Inport(4));

% APF: (Iabc, t, apf_on, Ts)
connect(mdl, plt.Outport(2),  apf.Inport(1));
connect(mdl, clk.Outport(1),  apf.Inport(2));
connect(mdl, apfE.Outport(1), apf.Inport(3));
connect(mdl, tsC.Outport(1),  apf.Inport(4));

% PQ: (Vabc, Isource, t, Ts)
connect(mdl, plt.Outport(1),  pq.Inport(1));
connect(mdl, apf.Outport(1),  pq.Inport(2));
connect(mdl, clk.Outport(1),  pq.Inport(3));
connect(mdl, tsC.Outport(1),  pq.Inport(4));

% Twin: (THD_i, Ppv, t, Ts)
connect(mdl, pq.Outport(2),   tw.Inport(1));
connect(mdl, plt.Outport(3),  tw.Inport(2));
connect(mdl, clk.Outport(1),  tw.Inport(3));
connect(mdl, tsC.Outport(1),  tw.Inport(4));

% Scopes & logs
connect(mdl, pq.Outport(1),   mTHD.Inport(1));
connect(mdl, pq.Outport(2),   mTHD.Inport(2));
connect(mdl, mTHD.Outport(1), ph('THD Scope').Inport(1));
connect(mdl, tw.Outport(1),   mEco.Inport(1));
connect(mdl, tw.Outport(2),   mEco.Inport(2));
connect(mdl, mEco.Outport(1), ph('Econ Scope').Inport(1));
connect(mdl, pq.Outport(2),   ph('log_THD_i').Inport(1));
connect(mdl, tw.Outport(1),   ph('log_cost').Inport(1));

try
    Simulink.BlockDiagram.arrangeSystem(mdl);
catch
    fprintf('[build] arrangeSystem unavailable (pre-R2018b) - skipping.\n');
end
save_system(mdl);

fprintf('\n==============================================================\n');
fprintf(' Model "%s.slx" built.\n', mdl);
fprintf(' Run:  sim(''%s'')\n', mdl);
fprintf(' Expected: THD_i ~27%% before t=%.2fs, <5%% after APF on.\n', P.apf_on_t);
fprintf('==============================================================\n');
end

% =========================================================================
%  HELPERS
% =========================================================================
function addb(path, src, pos)
add_block(src, path);
set_param(path,'Position',pos);
end

function connect(mdl, srcPort, dstPort)
add_line(mdl, srcPort, dstPort, 'autorouting','on');
end

function addMLfun(mdl, name, code, pos)
p = [mdl '/' name];
add_block('simulink/User-Defined Functions/MATLAB Function', p);
set_param(p,'Position',pos);
rt = sfroot;
ch = rt.find('-isa','Stateflow.EMChart','Path',p);
ch.Script = code;
end

% =========================================================================
%  EMBEDDED MATLAB-FUNCTION SOURCE
% =========================================================================
function c = plantCode()
L = {
'function [Vabc, Iload, Ppv, freq] = plant(t, G, T, sag)'
'%#codegen'
'f0 = 50; w0 = 2*pi*f0; Vpk = 230*sqrt(2);'
'sagFactor = 1 - 0.30*sag;'
'Va = sagFactor*Vpk*sin(w0*t);'
'Vb = sagFactor*Vpk*sin(w0*t - 2*pi/3);'
'Vc = sagFactor*Vpk*sin(w0*t + 2*pi/3);'
'Vabc = [Va; Vb; Vc];'
'Ifund = 15; ho = [5 7 11 13]; hm = [0.20 0.14 0.09 0.077];'
'Ia = Ifund*sin(w0*t);'
'Ib = Ifund*sin(w0*t - 2*pi/3);'
'Ic = Ifund*sin(w0*t + 2*pi/3);'
'for k = 1:numel(ho)'
'   n = ho(k); amp = Ifund*hm(k);'
'   Ia = Ia + amp*sin(n*w0*t);'
'   Ib = Ib + amp*sin(n*(w0*t - 2*pi/3));'
'   Ic = Ic + amp*sin(n*(w0*t + 2*pi/3));'
'end'
'Iload = [Ia; Ib; Ic];'
'Pstc = 10000; gam = -0.0040; NOCT = 45;'
'Tcell = T + (G/800)*(NOCT - 20);'
'Ppv = Pstc*(G/1000)*(1 + gam*(Tcell - 25));'
'if Ppv < 0, Ppv = 0; end'
'freq = f0;'
'end'
};
c = strjoin(L, char(10));
end

function c = apfCode()
L = {
'function Isource = apf(Iabc, t, apf_on, Ts)'
'%#codegen'
'persistent IdLP IqLP'
'if isempty(IdLP), IdLP = 0; IqLP = 0; end'
'f0 = 50; w0 = 2*pi*f0; th = w0*t;'
'Ia = Iabc(1); Ib = Iabc(2); Ic = Iabc(3);'
'Id =  (2/3)*( Ia*cos(th) + Ib*cos(th-2*pi/3) + Ic*cos(th+2*pi/3) );'
'Iq = -(2/3)*( Ia*sin(th) + Ib*sin(th-2*pi/3) + Ic*sin(th+2*pi/3) );'
'fc = 25; a = 2*pi*fc*Ts/(1 + 2*pi*fc*Ts);'
'IdLP = IdLP + a*(Id - IdLP);'
'IqLP = IqLP + a*(Iq - IqLP);'
'Idh = Id - IdLP; Iqh = Iq - IqLP;'
'Iah = Idh*cos(th)        - Iqh*sin(th);'
'Ibh = Idh*cos(th-2*pi/3) - Iqh*sin(th-2*pi/3);'
'Ich = Idh*cos(th+2*pi/3) - Iqh*sin(th+2*pi/3);'
'Icomp = [Iah; Ibh; Ich];'
'Isource = Iabc - apf_on*Icomp;'
'end'
};
c = strjoin(L, char(10));
end

function c = pqCode()
L = {
'function [THD_v, THD_i, Vrms, Irms] = pqanalyzer(Vabc, Isource, t, Ts)'
'%#codegen'
'N = 1000;'
'persistent vbuf ibuf idx thdv thdi vr ir'
'if isempty(vbuf)'
'   vbuf = zeros(N,1); ibuf = zeros(N,1); idx = 0;'
'   thdv = 0; thdi = 0; vr = 0; ir = 0;'
'end'
'idx = idx + 1;'
'vbuf(idx) = Vabc(1); ibuf(idx) = Isource(1);'
'if idx >= N'
'   vr = sqrt(mean(vbuf.^2));'
'   ir = sqrt(mean(ibuf.^2));'
'   thdv = localTHD(vbuf, N, Ts);'
'   thdi = localTHD(ibuf, N, Ts);'
'   idx = 0;'
'end'
'THD_v = thdv; THD_i = thdi; Vrms = vr; Irms = ir;'
'end'
''
'function thd = localTHD(x, N, Ts)'
'fs = 1/Ts;'
'df = fs/N;'
'fund_bin = round(50/df) + 1;'
'X = abs(fft(x))*(2/N);'
'fund = X(fund_bin);'
'hsum = 0;'
'for k = 2:25'
'   b = round(k*50/df) + 1;'
'   if b <= N/2, hsum = hsum + X(b)^2; end'
'end'
'if fund > 1e-6, thd = sqrt(hsum)/fund; else, thd = 0; end'
'end'
};
c = strjoin(L, char(10));
end

function c = twinCode()
L = {
'function [cost_BDT_yr, SoH_loss_pct, batt_Temp] = digitaltwin(THD_i, Ppv, t, Ts)'
'%#codegen'
'persistent SoHloss'
'if isempty(SoHloss), SoHloss = 0; end'
'Pbatt = max(Ppv,1)/1000;'
'dE = Pbatt*Ts/3600;'
'base_fade = 8e-7;'
'extra = 1 + THD_i^2; dT = 25*(extra-1);'
'accel = 2^(dT/10);'
'SoHloss = SoHloss + base_fade*dE*accel;'
'batt_Temp = 30 + dT;'
'SoH_loss_pct = SoHloss*100;'
'Batt_kWh = 20; cost_kWh = 24000;'
'yearScale = (365*24*3600)/max(t,Ts);'
'cost_BDT_yr = SoHloss*Batt_kWh*cost_kWh*yearScale;'
'end'
};
c = strjoin(L, char(10));
end
