# DL model for the Prima Indian data

# load packages
using Pkg
Pkg.status()
Pkg.build("Knet")
using Knet
using PyPlot
using LinearAlgebra
using Random
using Printf
import StatsBase.predict

# load help function
include(pwd()*"/utilities.jl")

Random.seed!(1234) # set random numbers

x_train,y_train,x_val,y_val,x_test,y_test = load_data([1,2],0.2)

# define network
n_input = 8
n_hidden_1 = 50
n_hidden_2 = 20
n_out = 2


# predict function

function predict(w,x,p)
    output = zeros(2,size(x,2))
    for i=1:2:length(w)-2
        if i > 1
            x = dropout(x,p)
        end
        x = relu.(w[i]*x .+ w[i+1])
    end
    return softmax(w[end-1]*x .+ w[end];dims=1)
end

loss(w,x,ygold,p, lambda) = Knet.nll(predict(w,x,p), ygold; average=true) + lambda[1]*sum(w[1].^2)  + lambda[2]*sum(w[3].^2)  + lambda[3]*sum(w[5].^2)

lossgradient = grad(loss)

function train(epoch::Int,dropout_percentage::Real,lambda::Vector, w, optim)

    loss_train = zeros(epoch)
    loss_val = zeros(epoch)

    for i in 1:epoch

        idx_train = rand(1:size(x_train,2),size(x_train,2))
        idx_val = rand(1:size(x_val,2),size(x_val,2))

        grads = lossgradient(w,x_train[:,idx_train[:]],y_train[idx_train],dropout_percentage,lambda)
        update!(w, grads, optim)

        loss_train[i] = loss(w,x_train[:,idx_train[:]], y_train[idx_train], 0,lambda)
        loss_val[i] = loss(w,x_val[:,idx_val], y_val[idx_val], 0,lambda)

    end

    return loss_train, loss_val

end
