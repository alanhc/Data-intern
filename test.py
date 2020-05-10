import utils.method as m


# python test.py best.joblib 35.187.144.113 5432 answer.csv
def main():
    solution = m.MLsolution()
    # handle data
    solution.get_data()
    solution.preprocessing_data()
    # make a X_test(feature) for model's input 
    X_test = solution.features
    
    solution.run_predict(X_test)
    
        
if __name__ == "__main__":
    main()