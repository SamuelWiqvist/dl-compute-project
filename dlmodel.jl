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


x_train,y_train,x_val,y_val,x_test,y_test = load_data([1,2])

# define network
n_input = 8
n_hidden_1 = 100
n_hidden_2 = 50
n_hidden_3 = 25
n_out = 2

w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
         xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
         xavier(n_hidden_3,n_hidden_2), zeros(n_hidden_3,1),
         xavier(n_out,n_hidden_3), zeros(n_out,1)]

#=
w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
         xavier(n_out,n_hidden_1), zeros(n_out,1)]
=#

nbr_training_obs = length(y_train)
nbr_parameters = 0


for i in w
    global nbr_parameters = nbr_parameters + size(i,1)*size(i,2)
end

@printf "Nbr training obs %d, nbr parameters %d, obs/parameters %.2f\n" nbr_training_obs nbr_parameters nbr_training_obs/nbr_parameters

# predict function
function predict(w,x)
    output = zeros(2,size(x,2))
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

loss(w,x,ygold) = Knet.nll(predict(w,x), ygold; average=false)

lossgradient = grad(loss)

optim = optimizers(w, Adam)

function train(epoch::Int,dropout_percentage::Real,lambda::Vector)

    loss_train = zeros(epoch)
    loss_val = zeros(epoch)

    idx_train = ones(Int, size(x_train,2))
    idx_val = ones(Int, size(x_val,2))
    first_idx = findfirst(y_val .== 1)
    idx_train[1] = size(x_train,2)
    idx_val[1] = first_idx

    for i in 1:epoch

        ixd_rand = rand(1:size(x_train,2)-1,size(x_train,2)-1)
        idx_train[2:end] = ixd_rand

        ixd_rand = rand(setdiff(1:size(x_val,2),first_idx),size(x_val,2)-1)
        idx_val[2:end] = ixd_rand

        grads = lossgradient(w,x_train[:,idx_train[:]],y_train[idx_train])
        update!(w, grads, optim)

        loss_train[i] = loss(w,x_train[:,idx_train[:]], y_train[idx_train])
        loss_val[i] = loss(w,x_val[:,idx_val], y_val[idx_val])

    end

    return loss_train, loss_val

end


# training
@time loss_train, loss_val = train(5000,0,zeros(3))

y_pred_train_proc = predict(w, x_train)
y_pred_proc = predict(w, x_test)

y_pred_train = zeros(length(y_train))
y_pred_test = zeros(length(y_test))

for i = 1:length(y_train)
    if findmax(y_pred_train_proc[:,i])[2]  == 1
        y_pred_train[i] = 1
    else
        y_pred_train[i] = 2
    end
end

for i = 1:length(y_test)
    if findmax(y_pred_proc[:,i])[2]  == 1
        y_pred_test[i] = 1
    else
        y_pred_test[i] = 2
    end
end

y_pred_train = class_nll(y_pred_train_proc,labels)
y_pred_test = class_nll(y_pred_proc,labels)


# loss
loss_training = loss(w,x_train[:,end:-1:1], y_train[end:-1:1])
loss_test = loss(w,x_test[:,end:-1:1], y_test[end:-1:1])

# classification results
class_res_training = classification_results(y_pred_train, y_train, [1,2])
class_res_test = classification_results(y_pred_test, y_test, [1,2])

# loss as function of epoch
PyPlot.figure()
PyPlot.plot(loss_train, "b")

PyPlot.figure()
PyPlot.plot(loss_val, "r")
