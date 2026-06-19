struct Config {
	float TimeSpan;

	//float PhotonEnergy;
	float PhotonLossRate;
	float DephasingRate;
	float EmissionRate;
	float PumpingRate;
	float CollectiveDephasingRate;
	float CollectiveEmissionRate;
	float CollectivePumpingRate;
	float CavityEmissionRate;
	float CavityAbsorptionRate;

	int TrajectoryCount;
	int RungeKuttaPoly;
	float JumpTolerance;
	float ShrinkTolerance;
};

