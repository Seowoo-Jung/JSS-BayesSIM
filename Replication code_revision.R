## library
library(BayesSIM)
library(np)
library(mgcv)
library(PLSiMCpp)
library(tgp)
library(plgp)
library(tidyverse)

# 4. Illustration
##########################
#### Data preparation ####
##########################
X <- DATA1$X; y <- DATA1$y
n <- nrow(X)
set.seed(123)
idx <- sample(1:n, round(n*0.7))
X_train <- X[idx,]; X_test <- X[-idx,]
y_train <- y[idx]; y_test <- y[-idx]
data_train <- data.frame(index = X_train, y = y_train)
data_test <- data.frame(index = X_test, y = y_test)

## 4.2 Model definition
# Set Prior
# Construction of prior parameters
prior <- list(index = list(direction = NULL, dispersion = 150),
              link = list(basis = list(df = 21, degree = 2, delta = 0.001),
                          beta = list(mu = NULL, cov = NULL)),
              sigma2 = list(shape = 0.001, rate = 100))

# Execution of prior parameters
prior <- prior.param(indexprior = "fisher", link = "bspline")


# Set initial values
# Construction
initialValues <- list(index = NULL,
                      link = list(beta = NULL),
                      sigma2 = 0.01)

# Execution
initialValues <- init.param(indexprior = "fisher", link = "bspline")

# Check monitorable MCMC arguments
fit_temp <- bsFisher.setup(y ~ ., data = data_train, setSeed = TRUE)
getVarMonitor(fit_temp, type = "name")
getVarMonitor(fit_temp, type = "list")
monitorVariable <- c("beta[1]", "beta[2]")
getModelDef(fit_temp)

# Fit model
## One tool version
fit <- BayesSIM(y ~ ., data = data_train,
                indexprior = "fisher", link = "bspline",
                nchain = 2, niter = 5000, nburnin = 1000, setSeed = TRUE)
# Split version
fit_model <- BayesSIM.setup(y ~ ., data = data_train,
                            indexprior = "fisher", link = "bspline",
                            nchain = 2, setSeed = TRUE)
Ccompile <- compileModelAndMCMC(models)
nimSampler <- get_sampler(Ccompile)
initList <- getInit(fit_model)
mcmc.out <- runMCMC(nimSampler, niter = 5000, nburnin = 1000, thin = 1,
                    nchains = 2, setSeed = TRUE, inits = initList,
                    summary = TRUE, samplesAsCodaMCMC = TRUE)

# 5. Post-processing
fit <- as.bsim(fit_model, mcmc.out)
## 5.1 Posterior summaries
summary(fit)
coef(fit, method = "mean", se = TRUE)
gof(fit)

## 5.2 Convergence diagnostics (Figure 1)
nimTraceplot(fit)



## 5.3 Fitted values
pred <- predict(fit, newdata = data_test,
                type = "response", method = "mean", se.fit = TRUE)
## Fitted plot fortest data
plot(pred)

## Fitted plot for train data
plot(fit)


## 5.4 Compare different models (Table 6)
# bsFisher (this!) -------------------------------------------------------------
fit_bsFisher <- BayesSIM(y ~ ., data = data_train, setSeed = 123)
index_bsFisher <- coef(fit_bsFisher)
indexSD_bsFisher <- coef(fit_bsFisher, se = TRUE)[2, ]
pred_bsFisher <- predict(fit_bsFisher,
                         newdata = data_test)

# bsSphere-------------------------------------------------------------
fit_bsSphere <- BayesSIM(y ~ ., data = data_train,
                         indexprior = "sphere", link = "bspline",
                         setSeed = 123)
index_bsSphere <- coef(fit_bsSphere)
indexSD_bsSphere <- coef(fit_bsSphere, se = TRUE)[2,]
pred_bsSphere <- predict(fit_bsSphere,
                         newdata = data_test)

# bsPolar-------------------------------------------------------------
fit_bsPolar <- BayesSIM(y ~ ., data = data_train,
                        indexprior = "polar", link = "bspline",
                        setSeed = 123)
index_bsPolar <- coef(fit_bsPolar)
indexSD_bsPolar <- coef(fit_bsPolar, se = TRUE)[2,]
pred_bsPolar <- predict(fit_bsPolar,
                        newdata = data_test)

# bsSpike-------------------------------------------------------------
fit_bsSpike <- BayesSIM(y ~ ., data = data_train,
                        indexprior = "spike", link = "bspline",
                        setSeed = 123)
index_bsSpike <- coef(fit_bsSpike)
indexSD_bsSpike <- coef(fit_bsSpike, se = TRUE)[2,]
pred_bsSpike <- predict(fit_bsSpike,
                        newdata = data_test)

# gpFisher ------------------------------------------------------------
fit_gpFisher <- BayesSIM(y ~ ., data = data_train,
                         indexprior = "fisher", link = "gp",
                         setSeed = 123)
index_gpFisher <- coef(fit_gpFisher)
indexSD_gpFisher <- coef(fit_gpFisher, se = TRUE)[2, ]
pred_gpFisher  <- predict(fit_gpFisher, newdata = data_test)

# gpSphereEG ----------------------------------------------------------
fit_gpSphere <- BayesSIM(y ~ ., data = data_train,
                         indexprior = "sphere", link = "gp",
                         niter = 700, nburnin = 200,
                         method = "EG", setSeed = 123)
index_gpSphere <- coef(fit_gpSphere)
indexSD_gpSphere <- coef(fit_gpSphere, se = TRUE)[2, ]
pred_gpSphere_  <- predict(fit_gpSphere, newdata = data_test)

# gpPolar -------------------------------------------------------------
fit_gpPolar <- BayesSIM(y ~ ., data = data_train,
                        indexprior = "polar", link = "gp",
                        setSeed = 123)
index_gpPolar <- coef(fit_gpPolar)
indexSD_gpPolar <- coef(fit_gpPolar, se = TRUE)[2, ]
pred_gpPolar_  <- predict(fit_gpPolar, newdata = data_test)

# gpSpike -------------------------------------------------------------
fit_gpSpike <- BayesSIM(y ~ ., data = data_train,
                        indexprior = "spike", link = "gp",
                        setSeed = 123)
index_gpSpike <- coef(fit_gpSpike)
indexSD_gpSpike <- coef(fit_gpSpike)[2, ]
pred_gpSpike_  <- predict(fit_gpSpike, newdata = data_test)

#########################
####### Results #########
#########################

modelName <- c("bsFisher","bsSphere","bsPolar","bsSpike",
               "gpFisher", "gpSphere","gpPolar","gpSpike")

rmse <- c(pred_bsFisher$rmse, pred_bsSphere$rmse, pred_bsPolar$rmse,
          pred_bsSpike$rmse, pred_gpFisher$rmse, pred_gpSphere_$rmse,
          pred_gpPolar_$rmse, pred_gpSpike_$rmse)

## Index matrix
index <- list(
  bsFisher = index_bsFisher,
  bsSphere = index_bsSphere,
  bsPolar  = index_bsPolar,
  bsSpike  = index_bsSpike,
  gpFisher = index_gpFisher,
  gpSphere = index_gpSphere,
  gpPolar  = index_gpPolar,
  gpSpike  = index_gpSpike
)

# SD
indexSD <- list(
  bsFisher = indexSD_bsFisher,
  bsSphere = indexSD_bsSphere,
  bsPolar  = indexSD_bsPolar,
  bsSpike  = indexSD_bsSpike,
  gpFisher = indexSD_gpFisher,
  gpSphere = indexSD_gpSphere,
  gpPolar  = indexSD_gpPolar,
  gpSpike  = indexSD_gpSpike
)

## summary table: mean (sd)  + cosfun + rmse
ix_cols_pretty <- paste("index", 1:4)
tab <- do.call(rbind, lapply(modelName, function(nm){

  mu <- index[[nm]]
  sd <- indexSD[[nm]]

  fmt <- function(a,b) sprintf("%.4f (%.4f)", a, b)
  row <- data.frame(
    modelName = nm,
    `index 1` = fmt(mu[1], sd[1]),
    `index 2` = fmt(mu[2], sd[2]),
    `index 3` = fmt(mu[3], sd[3]),
    `index 4` = fmt(mu[4], sd[4]),
    stringsAsFactors = FALSE, check.names = FALSE
  )

  # cos(angle)
  true_theta <- c(2, 1, 1, 1)/sqrt(7)
  mu_norm <- mu / sqrt(sum(mu^2))
  row$cosfun <- if (all(is.finite(mu_norm))) round(acos(sum(true_theta * mu_norm)), 4) else NA_real_

  # rmse
  row$rmse <- round(rmse[match(nm, modelName)], 4)
  row
}))

row.names(tab) <- NULL
colnames(tab) <- c("Model name", paste("index", 1:4, "(sd)"), "Angle", "RMSE")
tab
# write.csv(tab, "FINAL_illustration.csv")

################################################################################
################################################################################

# 6. Comparison with existing packages
## Table 7
##########################
#### Data preparation ####
##########################
set.seed(123)
n <- nrow(concrete)
p <- ncol(concrete)-1
concrete[,-9] <- scale(concrete[,-9])
idx <- sample(1:n, round(n*0.7))
X_train <- concrete[idx,-9]
X_test <- concrete[-idx,-9]
y_train <- concrete[idx, c("strength")]
y_test <- concrete[-idx, c("strength")]
df_DATA_train <- data.frame(X_train, y = y_train)
df_DATA_test <- data.frame(X_test, y = y_test)

#### Modeling ####
# np --------------------------------------------------------------------
start_np <- Sys.time()
model_np <- npindex(y ~ cement + blast_furnace_slag + flay_ash + water +
                      superplasticizer + coarse_aggreate + fine_aggregate + age,
                    data = df_DATA,
                    newdata = df_DATA_test,
                    nmulti = 1)
end_np <- Sys.time()
index_np <- model_np$beta
index_np <- index_np/sqrt(sum(index_np^2))
rmse_np <- sqrt(mean((fitted(model_np) - y_test)^2))
time_np <- end_np - start_np

# ppr --------------------------------------------------------------------
start_ppr <- Sys.time()
model_ppr <- ppr(y ~ cement + blast_furnace_slag + flay_ash + water +
                   superplasticizer + coarse_aggreate + fine_aggregate + age,
                 data = df_DATA, nterms = 1)
end_ppr <- Sys.time()
pred <- predict(model_ppr, newdata = df_DATA_test)
index_ppr <- model_ppr$alpha
rmse_ppr <- sqrt(mean((pred-y_test)^2))
time_ppr <- end_ppr - start_ppr

# mgcv -------------------------------------------------------------------
si <- function(theta, y, x, opt=TRUE, k=10, fx=FALSE) {
  ## Fit single index model using gam call, given theta (defines alpha).
  ## Return ML if opt==TRUE and fitted gam with theta added otherwise.
  ## Suitable for calling from 'optim' to find optimal theta/alpha.
  alpha <- c(1,theta) ## constrained alpha defined using free theta
  kk <- sqrt(sum(alpha^2))
  alpha <- alpha/kk  ## so now ||alpha||=1
  a <- x %*% alpha     ## argument of smooth
  b <- gam(y~s(a,fx=fx,k=k) - 1,family=gaussian,method="ML") ## fit model
  if (opt) return(b$gcv.ubre) else {
    b$alpha <- alpha  ## add alpha
    J <- outer(alpha,-theta/kk^2) ## compute Jacobian
    for (j in 1:length(theta)) J[j+1,j] <- J[j+1,j] + 1/kk
    b$J <- J ## dalpha_i/dtheta_j
    return(b)
  }
} ## si


start_mgcv <- Sys.time()
th0 <- rep(0, 7)
# get initial theta, using no penalization
f0 <- optim(th0, si, y = y_train, x = as.matrix(X_train), fx = TRUE, k = 5)
# now get theta/alpha with smoothing parameter selection...
f1 <- optim(f0$par, si, y = y_train, x = as.matrix(X_train), hessian = TRUE, k = 10)
theta.est <- f1$par
thest.std <- theta.est/sqrt(sum(theta.est^2))

## extract and examine fitted model...
b <- si(thest.std, y_train, x = as.matrix(X_train), opt=FALSE) ## extract best fit model
plot(b, pages=1)
b
index_mgcv <- b$alpha
end_mgcv <- Sys.time()
time_mgcv <- end_mgcv - start_mgcv

## prediction
X_test_matrix <- as.matrix(X_test)
eta_new <- X_test_matrix %*% index_mgcv
predict_data <- data.frame(a = eta_new)
predicted_response <- predict(b,
                              newdata = predict_data,
                              type = "response")
rmse_mgcv <- sqrt(mean((predicted_response - y_test)^2))

# PLSiMCpp -------------------------------------------------------------------
start_PLSiMCpp <- Sys.time()
fit_PLSim <- plsim.est(xdat = NULL, zdat = X_train, ydat = as.data.frame(y_train), TargetMethod = "plsimest",
                       ParmaSelMethod = "CrossValidation")
end_PLSiMCpp <- Sys.time()

# index
index_PLSiMCpp <- fit_PLSim$zeta[,1]
pred_PLSim <- predict(fit_PLSim, z_test = as.matrix(X_test))

# RMSE
rmse_PLSiMCpp <- sqrt(mean((pred_PLSim[,1] - y_test)^2))
# plot(pred_PLSim[,1], y_test)
# abline(0, 1, col = "red")
time_PLSiMCpp <- end_PLSiMCpp - start_PLSiMCpp
# tgp --------------------------------------------------------------------
start_tgp <- Sys.time()
model_tgp <- bgp(X = X_train,
                 Z = df_DATA$y,
                 XX = X_test,
                 BTE  = c(1000, 5000, 1),
                 meanfn = "constant",
                 corr = "sim", trace = TRUE)
end_tgp <- Sys.time()
pred_tgp <- model_tgp$ZZ.mean
index_samp <- t(apply(model_tgp$trace$XX[[1]][,paste0("d", 1:p)],
                      1, function(x) x/sqrt(sum(x^2))))
index_tgp_mean <- apply(index_samp, 2, mean)
index_tgp_sd <- apply(index_samp, 2, sd)

rmse_tgp <- sqrt(mean((pred_tgp-y_test)^2))
time_tgp <- end_tgp - start_tgp


# plgp --------------------------------------------------------------------
graphics.off()
formals(data.GP)$X <- as.matrix(df_DATA[,-9])
formals(data.GP)$Y <- as.matrix(df_DATA$y)

## default prior
prior <- prior.GP(p, "sim")
start <- ncol(X_train)/2
end <- nrow(X_train)

## Particle Learning Inference
start_plgp = Sys.time()
model_plgp <- PL(dstream=data.GP,
                 start = start, end = end,
                 init=draw.GP,
                 lpredprob=lpredprob.GP, propagate=propagate.GP,
                 prior=prior, addpall=addpall.GP,
                 params=params.GP, P = 50)
end_plgp = Sys.time()

# test data: posterior predictive
outp <- papply(XX = as.matrix(X_test), fun = pred.GP,
               Y=PL.env$pall$Y, quants = TRUE, prior=prior)
m <- rep(0, nrow(as.matrix(X_test)))
for(i in 1:length(outp)) m <- m + outp[[i]]$m
m <- m / length(outp)

## the mean and SD of the particles predictive
params <- params.GP()
indexSamp <- params[, paste0("d.", 1:p)]
indexSamp <- t(apply(indexSamp, 1, function(x){
  if(x[1] < 0){ x <- x *(-1)}
  else {x}
  return(x/sqrt(sum(x^2)))
}))
index_plgp <- apply(indexSamp, 2, mean)
indexSD_plgp <- apply(indexSamp, 2, sd)

## a calculation of RMSE to the truth
rmse_plgp <- sqrt(mean((m - y_test)^2))
time_plgp <- end_plgp - start_plgp


#### Result ####
modelName <- c("np", "ppr", "mgcv", "PLSiMCpp", "tgp", "plgp")

rmse <- c(rmse_np, rmse_ppr, rmse_mgcv, rmse_PLSiMCpp, rmse_tgp, rmse_plgp)
time <- c(time_np, time_ppr, time_mgcv, time_PLSiMCpp, time_tgp, time_plgp)
index_mean <- rbind(index_np,
                    index_ppr,
                    index_mgcv,
                    index_PLSiMCpp,
                    index_tgp_mean,
                    index_plgp)
index_sd <- rbind(index_tgp_sd,
                  indexSD_plgp)

p <- ncol(index_mean)
colnames(index_mean) <- paste0("index", 1:p)
colnames(index_sd)   <- paste0("index", 1:p)
rownames(index_mean) <- modelName
rownames(index_sd) <- c("tgp", "plgp")

fmt_num <- function(x, digits = 3) sprintf(paste0("%.", digits, "f"), x)
fmt_pair <- function(m, s, d_mean = 3, d_sd = 4) {
  paste0(fmt_num(m, d_mean), " (", fmt_num(s, d_sd), ")")
}

fmt_time_secs <- function(x, digits = 2, suffix = TRUE) {
  out <- sprintf(paste0("%.", digits, "f"), x)
  if (suffix) out <- paste0(out, " secs")
  out
}


index_fmt <- lapply(seq_len(p), function(j) {
  m <- index_mean[3:4, j]
  s <- index_sd[, j]
  fmt_pair(m, s, d_mean = 3, d_sd = 4)
}) %>% as.data.frame(stringsAsFactors = FALSE)


colnames(index_fmt) <- colnames(concrete)[-9]
colnames(index_mean) <- colnames(concrete)[-9]
index_fmt <- rbind(round(index_mean[1:2,], 3), index_fmt)

tbl <- data.frame(
  RMSE        = fmt_num(rmse, 4),
  `Total time`= fmt_time_secs(time, 4)
) %>%
  bind_cols(index_fmt)

rownames(tbl) <- modelName
out_csv <- cbind(model = rownames(tbl), tbl)
output_current <- as.data.frame(t(out_csv)[-1,])
# write.csv(output_current, "table_indices_current.csv", row.names = FALSE)

################################################################################
################################################################################
## Table 8

# set.seed(123)
# n <- nrow(concrete)
# concrete[,-9] <- scale(concrete[,-9])
# idx <- sample(1:n, round(n*0.7))
# X_train <- concrete[idx,-9]
# X_test <- concrete[-idx,-9]
# y_train <- concrete[idx, c("strength")]
# y_test <- concrete[-idx, c("strength")]
# df_DATA <- data.frame(X_train, y = y_train)
# df_DATA_test <- X_test
# df_DATA_test$y <- as.vector(y_test)

#### BayesSIM models ####
# bsFisher-------------------------------------------------------------
start <- Sys.time()
fit_bsFisher <- BayesSIM(y ~ ., data = df_DATA_train,
                         niter = 5000, nburnin = 1000,
                         setSeed = 123)
end <- Sys.time()
pred_bsFisher <- predict(fit_bsFisher,
                         newdata = df_DATA_test,
                         se.fit = TRUE)
index_bsFisher <- coef(fit_bsFisher)
indexSD_bsFisher <- coef(fit_bsFisher, se = TRUE)[2,]
time_bsFisher <- fit_bsFisher$input$time
totalTime_bsFisher <- end - start

# bsSphere-------------------------------------------------------------
start <- Sys.time()
fit_bsSphere <- BayesSIM(y ~ ., data = df_DATA_train,
                         indexprior = "sphere", link = "bspline",
                         niter = 5000, nburnin = 1000,
                         setSeed = 123)
end <- Sys.time()
pred_bsSphere <- predict(fit_bsSphere,
                         newdata = df_DATA_test,
                         se.fit = TRUE)
index_bseSphere <- coef(fit_bsSphere)
indexSD_bsSphere <- coef(fit_bsSphere, se = TRUE)[2,]
time_bsSphere <- fit_bsSphere$input$time
totalTime_bsSphere <- end - start

# bsPolar-------------------------------------------------------------
start <- Sys.time()
fit_bsplinePolar <- BayesSIM(y ~ ., data = df_DATA_train,
                             indexprior = "polar", link = "bspline",
                             niter = 5000, nburnin = 1000,
                             setSeed = 123)
end <- Sys.time()
pred_bsPolar <- predict(fit_bsplinePolar,
                        newdata = df_DATA_test,
                        se.fit = TRUE)
index_bsPolar <- coef(fit_bsPolar)
indexSD_bsPolar <- coef(fit_bsPolar, se = TRUE)[2,]
time_bsPolar <- fit_bsPolar$input$time
totalTime_bsPolar <- end - start

# bsSpike -------------------------------------------------------------
start <- Sys.time()
fit_bsSpike <- BayesSIM(y ~ ., data = df_DATA_train,
                        indexprior = "spike", link = "bspline",
                        niter = 5000, nburnin = 1000, setSeed = 123)
end <- Sys.time()
pred_bsSpike <- predict(fit_bsSpike,
                        newdata = df_DATA_test,
                        se.fit = TRUE)
index_bsSpike <- coef(fit_bsSpike)
indexSD_bsSpike <- coef(fit_bsSpike)[2,]
time_bsSpike <- fit_bsSpike$input$time
totalTime_bsSpike <- end - start

# gpFisher ------------------------------------------------------------
start <- Sys.time()
fit_gpFisher <- BayesSIM(y ~ ., data = df_DATA_train,
                         indexprior = "fisher", link = "gp",
                         niter = 5000, nburnin = 1000,
                         setSeed = 123)
end <- Sys.time()
pred_gpFisher  <- predict(fit_gpFisher, newdata = df_DATA_test, se.fit = TRUE)
index_gpFisher <- coef(fit_gpFisher)
indexSD_gpFisher <- coef(fit_gpFisher)[2,]
time_gpFisher <- fit_gpFisher$input$time
totalTime_gpFisher <- end - start

# gpSphereEB ----------------------------------------------------------
start <- Sys.time()
fit_gpSphereEB <- BayesSIM(y ~ ., data = df_DATA_train,
                           indexprior = "sphere", link = "gp",
                           niter = 5000, nburnin = 1000,
                           lowerB = c(rep(-1, p), -1e2, -1e2, -1e2),
                           upperB = c(rep(1, p), 1e2, 1e2, 1e2),
                           method = "EB", setSeed = 123)
end <- Sys.time()
pred_gpSphereEB  <- predict(fit_gpSphereEB, newdata = df_DATA_test, se.fit = TRUE)
index_gpSphereEB <- coef(fit_gpSphereEB)
indexSD_gpSphereEB <- coef(fit_gpSphereEB, se = TRUE)[2,]
time_gpSphereEB <- fit_gpSphereEB$input$time$samp
totalTime_gpSphereEB <- end - start

# gpPolar -------------------------------------------------------------
start <- Sys.time()
fit_gpPolarHigh <- BayesSIM(y ~ ., data = df_DATA_train,
                            indexprior = "polar", link = "gp",
                            niter = 5000, nburnin = 1000,
                            setSeed = 123)
end <- Sys.time()
pred_gpPolarHigh  <- predict(fit_gpPolarHigh, newdata = df_DATA_test, se.fit = TRUE)
index_gpPolarHigh <- coef(fit_gpPolarHigh)
indexSD_gpPolarHigh <- coef(fit_gpPolarHigh)[2,]
time_gpPolarHigh <- fit_gpPolarHigh$input$time
totalTime_gpPolarHigh <- end - start

# gpSpike -------------------------------------------------------------
start <- Sys.time()
fit_gpSpike <- BayesSIM(y ~ ., data = df_DATA_train,
                        indexprior = "spike", link = "gp",
                        niter = 5000, nburnin = 1000,
                        setSeed = 123)
end <- Sys.time()
pred_gpSpike_  <- predict(fit_gpSpike, newdata = df_DATA_test, se.fit = TRUE)
index_gpSpike <- coef(fit_gpSpike)
indexSD_gpSpike <- coef(fit_gpSpike, se = TRUE)
time_gpSpike <- fit_gpSpike$input$time
totalTime_gpSpike <- end - start

#### Results ####
modelName <- c("bsFisher","bsSphere","bsPolar","bsSpike",
               "gpFisher","gpSphere","gpPolar","gpSpike")

rmse <- c(pred_bsFisher$rmse, pred_bsSphere$rmse, pred_bsPolar$rmse, pred_bsSpike$rmse,
          pred_gpFisher$rmse, pred_gpSphereEB$rmse, pred_gpPolarHigh$rmse, pred_gpSpike_$rmse)

time <- c(time_bsFisher, time_bsSphere, time_bsPolar, time_bsSpike,
          time_gpFisher, time_gpSphereEB, time_gpPolarHigh, time_gpSpike)

totaltime <- c(totalTime_bsFisher, totalTime_bsSphere, totalTime_bsPolar, totalTime_bsSpike,
               totalTime_gpFisher, totalTime_gpSphereEB, totalTime_gpPolarHigh, totalTime_gpSpike)


index_mean <- rbind(index_bsFisher,
                    index_bsSphere,
                    index_bsPolar,
                    index_bsSpike,
                    index_gpFisher,
                    index_gpSphereEB,
                    index_gpPolarHigh,
                    index_gpSpike)

index_sd <- rbind(indexSD_bsFisher,
                  indexSD_bsSphere,
                  indexSD_bsPolar,
                  indexSD_bsSpike,
                  indexSD_gpFisher,
                  indexSD_gpSphereEB,
                  indexSD_gpPolarHigh,
                  indexSD_gpSpike)

p <- ncol(index_mean)
colnames(index_mean) <- paste0("index", 1:p)
colnames(index_sd)   <- paste0("index", 1:p)
rownames(index_mean) <- rownames(index_sd) <- modelName

fmt_num <- function(x, digits = 3) sprintf(paste0("%.", digits, "f"), x)

fmt_pair <- function(m, s, d_mean = 3, d_sd = 4) {
  paste0(fmt_num(m, d_mean), " (", fmt_num(s, d_sd), ")")
}

fmt_time_secs <- function(x, digits = 2, suffix = TRUE) {
  out <- sprintf(paste0("%.", digits, "f"), x)
  if (suffix) out <- paste0(out, " secs")
  out
}

index_fmt <- lapply(seq_len(p), function(j) {
  m <- index_mean[, j]
  s <- index_sd[, j]
  fmt_pair(m, s, d_mean = 3, d_sd = 4)
}) %>% as.data.frame(stringsAsFactors = FALSE)

names(index_fmt) <- colnames(concrete)[-9]

tbl <- data.frame(
  RMSE        = fmt_num(rmse, 4),
  `Total time`= fmt_time_secs(totaltime*60, 2),
  `Exe. time` = fmt_time_secs(time, 2)
) %>%
  bind_cols(index_fmt)

rownames(tbl) <- modelName
out_csv <- cbind(model = rownames(tbl), tbl)
output_bayesSIM <- as.data.frame(t(out_csv)[-1,])
# write.csv(output_bayesSIM, "table_indices_BayesSIM.csv", row.names = FALSE)
