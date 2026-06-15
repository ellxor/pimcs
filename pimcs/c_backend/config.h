struct Config {
	//int NumberOfEmitters;
	//int CavityTruncation;
	//int ExpectationOps;
	float TimeSpan;

	float PhotonEnergy;
	float PhotonLossRate;
	float DephasingRate;
	float EmissionRate;
	float PumpingRate;
	float CollectiveDephasingRate;
	float CollectiveEmissionRate;
	float CollectivePumpingRate;
	float CavityEmissionRate;
	float CavityAbsorptionRate;

	//int ThreadCount;
	int TrajectoryCount;
	int RungeKuttaPoly;
	float JumpTolerance;
	float ShrinkTolerance;
};

