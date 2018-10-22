# find hyperparameter values using gird search

include(pwd()*"/dlmodel.jl")

using Distributions
using ProgressMeter

N = 1000

dist = Uniform(0, 0.1)

rand(dist, 3)

hyperparameters = zeros(N,3)
model_fit_regularization = zeros(N,3+3+1+1)
model_fit_train = zeros(N,2000)
model_fit_val = zeros(N,2000)

# find hyperparameters using random search
@time begin
@showprogress for i in 1:N

    Random.seed!(1234) # set random numbers

    lambda = rand(dist,3)

    hyperparameters[i,:] = lambda

    w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
             xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
             xavier(n_out,n_hidden_2), zeros(n_out,1)]

    optim = optimizers(w, Adam)

    dropout_percentage = 0
    loss_train_vec, loss_val_vec = train(2000, dropout_percentage, lambda, w, optim)

    model_fit_train[i,:] = loss_train_vec
    model_fit_val[i,:] = loss_val_vec

    # calc predictions
    y_pred_train_proc = predict(w, x_train, 0)
    y_pred_val_proc = predict(w, x_val, 0)

    y_pred_train = class_nll(y_pred_train_proc,[1,2])
    y_pred_val = class_nll(y_pred_val_proc,[1,2])

    # loss
    model_fit_regularization[i,end-1] = loss(w,x_train, y_train, 0, lambda)
    model_fit_regularization[i,end] = loss(w,x_val, y_val, 0, lambda)

    # classification results
    model_fit_regularization[i,1:3] = classification_results(y_pred_train, y_train, [1,2])
    model_fit_regularization[i,4:6] = classification_results(y_pred_val, y_val, [1,2])

end
end

# add run w.o regularization


w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
         xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
         xavier(n_out,n_hidden_2), zeros(n_out,1)]

optim = optimizers(w, Adam)

# training
dropout_percentage = 0.
lambda = [0.0,0.0,0.0]
@time loss_train_vec, loss_val_vec = train(2000, dropout_percentage, lambda, w, optim)

# calc predictions
y_pred_train_proc = predict(w, x_train, 0)
y_pred_val_proc = predict(w, x_val, 0)

y_pred_train = class_nll(y_pred_train_proc,[1,2])
y_pred_val = class_nll(y_pred_val_proc,[1,2])

# loss
training_loss_no_reg = loss(w,x_train, y_train, 0, lambda)
val_loss_no_reg = loss(w,x_val, y_val, 0, lambda)

# classification results
training_classification_res_no_reg = classification_results(y_pred_train, y_train, [1,2])
val_classification_res_no_reg = classification_results(y_pred_val, y_val, [1,2])

# find best hyperparameter settings
(~, idx_best_training_loss) = findmin(model_fit_regularization[:,end-1])
(~, idx_best_val_loss) = findmin(model_fit_regularization[:,end])

(~, idx_best_training_tp) = findmax(model_fit_regularization[:,2])
(~, idx_best_val_tp) = findmax(model_fit_regularization[:,5])

(~, idx_best_training_tn) = findmax(model_fit_regularization[:,3])
(~, idx_best_val_tn) = findmax(model_fit_regularization[:,6])

(~, idx_best_training_acc) = findmax(model_fit_regularization[:,1])
(~, idx_best_val_acc) = findmax(model_fit_regularization[:,4])


@printf "Best lambda_1 %.10f\n" lambda_1[1]
@printf "Best lambda_2 %.10f\n" lambda_2[4]
@printf "Best lambda_3 %.10f\n" lambda_3[1]
