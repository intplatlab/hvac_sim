# hvac_sim

HVAC Simulink simulation

RL기반 HVAC control
- hvac-rltrn.py  RL 학습 코드
- data_loader.py	모델을 학습하기 위한 GPS DATA등을 DATA파일 중 data.txt에서 가져오는 코드

MATLAB/Simulink 건물공조시뮬레이션
- SIMULINK_Run.m	건물 공조 모델 시뮬레이션을 실행시키는 메인 코드
- usePython.m	Simulink내의 action_logic에서 Python RL 학습 코드와 통신하기 위한 코드 (내부에서 matpy.py호출)
- matpy.py	 Socket 통신 수행
