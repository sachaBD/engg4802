library(forecast)
library(forecTheta)

print('Loading data:')

df <- read.csv('data/filter_data.csv')

results <- matrix(0, 86258, 60)

print('Computing forecasts')
for (i in seq(0, 86258 - 400, 1)) # 17251
{    
    tryCatch(
            {
                out <- dotm(ts(df[1+i:400+i, 1]), h=60)
                results[i, 1:60] <- out$mean
            },
            error=function(error_message) {
            message("Error on theta")
            i = i + 20
            return(-999999)
            })

    if (i %% 10 == 0)
    {
        cat("Progress: ", i, " / 86258\n")
    }
    if (i %% 2000 == 0)
    {
        print("Saving results to: results/r_theta_results.csv")
        write.csv(results, 'results/r_theta_results.csv')
    }
}

print("Saving results to: results/r_theta_results.csv")
write.csv(results, 'results/r_theta_results.csv')