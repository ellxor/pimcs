struct Config {
	double PhotonLossRate;
	double DephasingRate;
	double EmissionRate;
	double PumpingRate;
	double CollectiveDephasingRate;
	double CollectiveEmissionRate;
	double CollectivePumpingRate;
	double CavityEmissionRate;
	double CavityAbsorptionRate;

	double StartTime;
	double EndTime;
	int TrajectoryCount;
	int RungeKuttaPoly;
	double JumpTolerance;
	double ShrinkTolerance;

	double InitialJ;
	double InitialM;
};

