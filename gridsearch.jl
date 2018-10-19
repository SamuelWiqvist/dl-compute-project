# find hyperparameter values using gird search

include(pwd()*"/dlmodel.jl")

p = [0]

lambda_1 = collect(0:0.02:0.5) 
lambda_2 = collect(0:0.02:0.5)
lambda_3 = [0]


model_fit_regularization = zeros(length(p),length(lambda_1),length(lambda_2),length(lambda_3),3+3+1+1)
model_fit_train = zeros(length(p),length(lambda_1),length(lambda_2),length(lambda_3),2000)
model_fit_val = zeros(length(p),length(lambda_1),length(lambda_2),length(lambda_3),2000)

# find hyperparameters using gridsearch
@time for i in 1:length(p)
    for j in 1:length(lambda_1)
        @printf "Percentage done %.2f\n" (j-1)/length(lambda_1)
        for k in 1:length(lambda_2)
            for l in 1:length(lambda_3)

                Random.seed!(1234) # set random numbers

                w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
                         #xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
                         xavier(n_out,n_hidden_1), zeros(n_out,1)]

                optim = optimizers(w, Adam)

                dropout_percentage = p[i]
                lambda = [lambda_1[j], lambda_2[k]]
                loss_train_vec, loss_val_vec = train(2000, dropout_percentage, lambda, w, optim)

                model_fit_train[i,j,k,l,:] = loss_train_vec
                model_fit_val[i,j,k,l,:] = loss_val_vec

                # calc predictions
                y_pred_train_proc = predict(w, x_train, 0)
                y_pred_val_proc = predict(w, x_val, 0)

                y_pred_train = class_nll(y_pred_train_proc,[1,2])
                y_pred_val = class_nll(y_pred_val_proc,[1,2])

                # loss
                model_fit_regularization[i,j,k,l,end-1] = loss(w,x_train, y_train, 0, lambda)/length(y_train)
                model_fit_regularization[i,j,k,l,end] = loss(w,x_val, y_val, 0, lambda)/length(y_val)

                # classification results
                model_fit_regularization[i,j,k,l,1:3] = classification_results(y_pred_train, y_train, [1,2])
                model_fit_regularization[i,j,k,l,4:6] = classification_results(y_pred_val, y_val, [1,2])

            end
        end
    end
end

# results without regularization
training_loss_no_reg = model_fit_regularization[1,1,1,1,end-1]
test_loss_no_reg = model_fit_regularization[1,1,1,1,end]
training_acc_no_reg = model_fit_regularization[1,1,1,1,1]
test_acc_no_reg = model_fit_regularization[1,1,1,1,4]

# find best hyperparameter settings
(~, idx_best_training_loss) = findmin(model_fit_regularization[:,:,:,:,end-1])
(~, idx_best_val_loss) = findmin(model_fit_regularization[:,:,:,:,end])

(~, idx_best_training_tp) = findmax(model_fit_regularization[:,:,:,:,2])
(~, idx_best_val_tp) = findmax(model_fit_regularization[:,:,:,:,5])

(~, idx_best_training_tn) = findmax(model_fit_regularization[:,:,:,:,3])
(~, idx_best_val_tn) = findmax(model_fit_regularization[:,:,:,:,6])

(~, idx_best_training_acc) = findmax(model_fit_regularization[:,:,:,:,1])
(~, idx_best_val_acc) = findmax(model_fit_regularization[:,:,:,:,4])

# 5,1,1,5
# 3,1,8,1

@printf "Best dropout percentage %.2f\n" p[1]
@printf "Best lambda_1 %.10f\n" lambda_1[1]
<<<<<<< HEAD
@printf "Best lambda_2 %.10f\n" lambda_2[4]
@printf "Best lambda_3 %.10f\n" lambda_3[1]
=======
@printf "Best lambda_2 %.10f\n" lambda_2[1]
@printf "Best lambda_3 %.10f\n" lambda_3[11]
>>>>>>> 2layernetwork
