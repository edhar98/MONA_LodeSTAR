find -L models -name "*Rod_weights.pth" | while read -r model_path; do
    echo "Running $model_path"
    python src/test_single_particle.py --model $model_path --config "$(dirname "$model_path")/config.yaml"
done
