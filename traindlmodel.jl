# train DL and clac predictions

include(pwd()*"/dlmodel.jl")

Random.seed!(1234) # set random numbers


w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
         xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
         xavier(n_out,n_hidden_2), zeros(n_out,1)]

nbr_training_obs = length(y_train)
nbr_parameters = 0


for i in w
    global nbr_parameters = nbr_parameters + size(i,1)*size(i,2)
end

@printf "Nbr training obs %d, nbr parameters %d, obs/parameters %.2f\n" nbr_training_obs nbr_parameters nbr_training_obs/nbr_parameters

optim = optimizers(w, Adam)

# training
dropout_percentage = 0.
lambda = [0.1,0.1, 0.1]
@time loss_train_vec, loss_val_vec = train(2000, dropout_percentage, lambda, w, optim)

# calc predictions
y_pred_train_proc = predict(w, x_train, 0)
y_pred_val_proc = predict(w, x_val, 0)
y_pred_proc = predict(w, x_test, 0)

y_pred_train = class_nll(y_pred_train_proc,[1,2])
y_pred_val = class_nll(y_pred_val_proc,[1,2])
y_pred_test = class_nll(y_pred_proc,[1,2])

# loss
loss_training = loss(w,x_train, y_train, 0, lambda)
loss_val = loss(w,x_val, y_val, 0, lambda)
loss_test = loss(w,x_test, y_test, 0, lambda)

# classification results
class_res_training = classification_results(y_pred_train, y_train, [1,2])
class_res_val = classification_results(y_pred_val, y_val, [1,2])
class_res_test = classification_results(y_pred_test, y_test, [1,2])


# loss as function of epoch
PyPlot.figure()
PyPlot.plot(loss_train_vec, "b")
PyPlot.ylabel("Loss")
PyPlot.xlabel("Epoch")

PyPlot.figure()
PyPlot.plot(loss_val_vec, "r")
PyPlot.ylabel("Loss")
PyPlot.xlabel("Epoch")
