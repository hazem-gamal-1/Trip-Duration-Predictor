from sklearn.metrics import r2_score,mean_squared_error

def evalute(model,X,Y_true,type):
    Y_pred=model.predict(X)
    r2=r2_score(Y_true,Y_pred)
    rmse=mean_squared_error(Y_true,Y_pred,squared=False)
    metrics = {
        "R2": float(r2),
        "RMSE": float(rmse),
    }
    print()
    print(f"{type} R2: {r2:.4f}, {type} RMSE: {rmse:.4f}")
    return metrics




def create_config_and_metrics(args, hyperparameters):
    config_and_metrics = {
        "model_configuration": {
            "poly_degree":args.poly_degree,
            "preprocessing": ["MinMaxScaler", "StandardScaler"][args.preprocessor - 1],
            "model": ["Ridge","MLPRegressor"][args.model - 1],
            "hyperparameters": hyperparameters,
        },
        "metrics": {
            "training": {},
            "Validation": {}
        }
    }
    return config_and_metrics
