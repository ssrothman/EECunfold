for nboot in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
do
    echo $nboot
    python scripts/build_reco.py Apr_23_2025 Pythia_inclusive --max_nboot $nboot &
done
wait
