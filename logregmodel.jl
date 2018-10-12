# Logistic regression model for the Prima Indian data

# load packages
using Pkg
using PyPlot
using LinearAlgebra
using Random
using Printf
using GLM
using DataFrames

# include files
include(pwd()*"/utilities.jl")

# load data
x_train,y_train,x_val,y_val,x_test,y_test = load_data()

# create data frame
training_data = zeros(size(x_train,2), size(x_train,1)+1)
training_data[:,1:end-1] = x_train'
training_data[:,end] = y_train
train_df = DataFrame(training_data)
test_df = DataFrame(x_test')

model = @formula(x9 ~ x1+x2+x3+x4+x5+x6+x7+x8)

@time logisticregresson = glm(model, train_df, Binomial(), LogitLink())

print(logisticregresson)

# compute predictions
y_pred_train = zeros(2,length(y_train))
y_pred_train[1,:] = predict(logisticregresson)
y_pred_train[2,:] = 1 .- y_pred_train[1,:]
y_pred_train = class(y_pred_train)

y_pred = zeros(2,length(y_test))
y_pred[1,:] = predict(logisticregresson,test_df)
y_pred[2,:] = 1 .- y_pred[1,:]
y_pred = class(y_pred)

# check classification results
class_res_training = classification_results(y_pred_train, y_train)
class_res_test = classification_results(y_pred, y_test)
