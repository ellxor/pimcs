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

	float TimeSpan;
	int TrajectoryCount;
	int RungeKuttaPoly;
	float JumpTolerance;
	float ShrinkTolerance;
};

