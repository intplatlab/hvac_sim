s = 200;
hour = 1;
minuite = 1;
day_to_sec = 3600 * 24;

% months = [1, 2, 3];
% days = [12, 1, 2, 21, 28, 1];
% hours = [19, 19, 19, 15, 15, 15];
% init_temps = [26, 26, 26, 27, 27, 27];
% 
% a = randi([280 280], 1);
% da = randi([20 20], 1);

Initial_Temp = 273 + (27);
mdl = 'building_HVAC_V00_027_AI_2019b';
in = Simulink.SimulationInput(mdl);

%6일 학습
% for train = 1:1
%     for step = 1:s
%         for daya = 1:6
%             %%%%%%%%%%%%%%%%%%%%%% one day code %%%%%%%%%%%%%%%%%%%%%%%%%%
%             %train_month = 1;
%             %da = 10;
%             %day_idx = 1;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%%%%%%%% many day code %%%%%%%%%%%%%%%%%%%%%%%%%
%             mon_idx = mod(daya, 3) + 1;
%             day_idx = mod(daya, 6) + 1;
%             train_months = months(mon_idx);
%             da = days(day_idx)-1;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Initial_Temp = 273 + 27;
%             start_time = hours(day_idx) * 3600 + da * day_to_sec;
%             init_time = start_time + 60;      
%             stop_time = start_time + 3697;
%             % 3697/5155
%             in = in.setModelParameter('StartTime',string(start_time),'StopTime',string(stop_time));
%             out = sim(in);
%             [train step daya]
%         end
%     end 
% end

% %하루 학습
% for train = 1:1
%     for step = 1:s
%         kkk = mod(step, 17) + 1;
%         Initial_Temp = 273 + 27;
%         start_time = hours(day_idx) * 3600 + da * day_to_sec;
%         init_time = start_time + initial_time(kkk);      
%         stop_time = start_time + time_delay(kkk);
%         in = in.setModelParameter('StartTime',string(start_time),'StopTime',string(stop_time));
%         out = sim(in);
%         [train step]
%     end
% end
% % 

%3일 학습
train_months = 1;
days = [20, 21, 22];
hours = 12;
for train = 1:1 
    for daya = 1:s
        for step = 1:3
            Initial_Temp = 273 + 28;
            da = days(step)-1;
            start_time = 12 * 3600 + da * day_to_sec;
            init_time = start_time +60;      
            stop_time = start_time + 3697;
            in = in.setModelParameter('StartTime',string(start_time),'StopTime',string(stop_time));
            out = sim(in);
            [daya, step]
        end
    end
end

% %3일 학습
% train_months = 1;
% days = [20, 21, 22];
% hours = 12;
% for train = 1:1
%     for daya = 1:3
%         for step = 1:s
%             Initial_Temp = 273 + 28;
%             da = days(daya)-1;
% %           da = days(day_idx+1)-1;
%             start_time = 12 * 3600 + da * day_to_sec;
%             init_time = start_time +60;      
%             stop_time = start_time + 3697;
%             in = in.setModelParameter('StartTime',string(start_time),'StopTime',string(stop_time));
%             out = sim(in);
% 
%         end
%     end
% end
