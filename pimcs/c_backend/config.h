struct Config {
	float PhotonLossRate;
	float DephasingRate;
	float EmissionRate;
	float PumpingRate;
	float CollectiveDephasingRate;
	float CollectiveEmissionRate;
	float CollectivePumpingRate;
	float CavityEmissionRate;
	float CavityAbsorptionRate;

	float StartTime;
	float EndTime;
	int TrajectoryCount;
	int RungeKuttaPoly;
	float JumpTolerance;
	float ShrinkTolerance;
};

