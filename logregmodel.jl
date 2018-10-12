# Logistic regression model for the Prima Indian data


include(pwd()*"/utilities.jl")

x_train,y_train,x_val,y_val,x_test,y_test = load_data()
