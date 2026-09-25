# Configuration and Hyperparameters for NBA Draft Prediction

# Data Split Configuration
VALIDATION_YEARS = [2022, 2023]
TEST_YEARS = [2024, 2025]

# Preprocessing & Feature Selection Options
IMPUTATION_STRATEGY = "median"  # "mean" or "median"
SCALING_METHOD = "standard"     # "standard" or "minmax"
SELECT_TOP_FEATURES = 40         # Number of top correlated features to keep (prevents overfitting). Set to None to use all.

# Target Transformation
# Options:
# - "linear": target = (61 - Pick) / 60
# - "reciprocal": target = 1.0 / Pick
# - "exponential": target = P_TARGET ** (Pick - 1)
# Note: Undrafted is always mapped to 0.0
TARGET_TYPE = "linear"
P_TARGET = 0.95                 # Base for exponential target score

# Neural Network Architecture
HIDDEN_SIZES = [32,16]             # Dimensions of hidden layers
ACTIVATION = "leaky_relu"       # "relu", "leaky_relu", "tanh", "sigmoid"
LEAKY_RELU_ALPHA = 0.01

# Training Hyperparameters
OPTIMIZER = "adam"              # "adam" or "sgd"
LEARNING_RATE = 0.01            # Learning rate
WEIGHT_DECAY = 0.0              # L2 regularization strength
DROPOUT_RATE = 0.2              # Dropout probability for hidden layers
BATCH_SIZE = 64
EPOCHS = 150

# Model Selection Checkpointing
# Options: 
# - "loss": saves the best model based on validation loss (MSE). More stable and prevents overfitting.
# - "rbo": saves the best model based on validation RBO. Highly sensitive to top-pick ranking, higher variance.
CHECKPOINT_METRIC = "loss"

# Loss Function Weighting (Only used if LOSS_TYPE is "mse")
LOSS_WEIGHT_FACTOR = 10.0      # Weights drafted players more heavily to focus RBO on top picks

# Training Objective Loss Type
# Options:
# - "mse": standard Mean Squared Error regression against target scores.
# - "siamese": pairwise RankNet training comparing player pairs within the same draft year.
# - "hybrid": trains both models and averages their predictions for inference.
LOSS_TYPE = "siamese"

# Siamese Network Hyperparameters
SIAMESE_SCALE = 1.0             # Scale/temperature factor inside the pairwise sigmoid
MAX_PAIRS_PER_YEAR = 1500       # Max player pairs sampled per draft class per epoch

# Ensemble Configuration
ENSEMBLE_SEEDS = [42, 107, 999, 7, 88]

# RBO Evaluation Parameter
P_RBO = 0.9                     # Persistence parameter for Rank Biased Overlap (higher = deeper look)

# Age-Based Post-processing Hyperparameters (Config B)
YOUNG_AGE_THRESHOLD = 19
YOUNG_AGE_MULTIPLIER = 1.3
YOUNG_SCORE_EXPONENT = 2.0       # 0.0 for flat multiplier, > 0.0 for score-dependent quadratic boost
OLD_AGE_THRESHOLD = 21
OLD_AGE_MULTIPLIER = 0.7
OLD_SCORE_EXPONENT = 0.0         # 0.0 for flat multiplier, > 0.0 for score-dependent penalty
EXEMPT_FORWARDS = True           # Exempts position 'F' (Forwards) from the age penalty


