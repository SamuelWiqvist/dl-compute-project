# find hyperparameter values using gird search

include(pwd()*"/dlmodel.jl")

p = collect(0:0.2:1)

lambda_1 = collect(0.00001:0.02:0.1)
lambda_2 = collect(0.00001:0.02:0.1)
lambda_3 = collect(0.00001:0.02:0.1)

trainin_val_res = zeros(length(p),length(lambda_1),length(lambda_2),length(lambda_3),3+3+1+1)

Random.seed!(1234) # set random numbers

@time for i in 1:length(p)
    @printf "Percentage done %.2f\n" (i-1)/length(p)
    for j in 1:length(lambda_1)
        for k in 1:length(lambda_2)
            for l in 1:length(lambda_3)

                w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
                         xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
                         xavier(n_out,n_hidden_2), zeros(n_out,1)]

                dropout_percentage = p[i]
                lambda = [lambda_1[j], lambda_2[k], lambda_3[l]]
                loss_train, loss_val = train(5000, dropout_percentage, lambda)

                # calc predictions
                y_pred_train_proc = predict(w, x_train, 0)
                y_pred_val_proc = predict(w, x_val, 0)

                y_pred_train = class_nll(y_pred_train_proc,[1,2])
                y_pred_val = class_nll(y_pred_val_proc,[1,2])

                # loss
                trainin_val_res[i,j,k,l,end-1] = loss(w,x_train, y_train, 0, lambda)/length(y_train)
                trainin_val_res[i,j,k,l,end] = loss(w,x_val, y_val, 0, lambda)/length(y_val)

                # classification results
                trainin_val_res[i,j,k,l,1:3] = classification_results(y_pred_train, y_train, [1,2])
                trainin_val_res[i,j,k,l,4:6] = classification_results(y_pred_val, y_val, [1,2])

            end
        end
    end
end

trainin_val_res

# loop over all combinations find best combination of hyperparameters 
