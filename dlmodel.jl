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

x_train,y_train,x_val,y_val,x_test,y_test = load_data()

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

nbr_training_obs = length(y_train)
nbr_parameters = 0


for i in w
    global nbr_parameters = nbr_parameters + size(i,1)*size(i,2)
end

@printf "Nbr training obs %d, nbr parameters %d, obs/parameters %.2f\n" nbr_training_obs nbr_parameters nbr_training_obs/nbr_parameters

# predict function
function predict(w,x)
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return sigm.(w[end-1]*x .+ w[end]) #sigm.(w[end-1]*x .+ w[end])
end

loss(w,x,ygold) = Knet.nll(predict(w,x), ygold)

lossgradient = grad(loss)

optim = optimizers(w, Adam)


# train network
epoch = 10000
loss_train = zeros(epoch)
loss_val = zeros(epoch)

idx_train = ones(Int, size(x_train,2))
idx_val = ones(Int, size(x_val,2))
first_idx = findfirst(y_val .== 1)
idx_train[1] = size(x_train,2)
idx_val[1] = first_idx

@time for i in 1:epoch

    ixd_rand = rand(1:size(x_train,2)-1,size(x_train,2)-1)
    idx_train[2:end] = ixd_rand

    ixd_rand = rand(setdiff(1:size(x_val,2),first_idx),size(x_val,2)-1)
    idx_val[2:end] = ixd_rand

    grads = lossgradient(w,x_train[:,idx_train[:]],y_train[idx_train])
    update!(w, grads, optim)

    loss_train[i] = loss(w,x_train[:,idx_train[:]], y_train[idx_train])
    loss_val[i] = loss(w,x_val[:,idx_val], y_val[idx_val])

end

# calc predicss
y_pred_train = class(predict(w, x_train))
y_pred = class(predict(w, x_test))

# loss
loss_training = loss(w,x_train[:,end:-1:1], y_train[end:-1:1])
loss_test = loss(w,x_test[:,end:-1:1], y_test[end:-1:1])

# classification results
class_res_training = classification_results(y_pred_train, y_train)
class_res_test = classification_results(y_pred, y_test)

# loss as function of epoch
PyPlot.figure()
PyPlot.plot(loss_train, "b")

PyPlot.figure()
PyPlot.plot(loss_val, "r")
