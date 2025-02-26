library(ggplot2)
library(tidyr)

Excess_Risk = data.frame('PENN' = c(0.12240982055664062, 0.021648883819580078, 0.06040370464324951, 0.08410537242889404, 0.005516409873962402, 0.08246016502380371, 0.0875016450881958, 0.047338128089904785, 0.0698162317276001, 0.16812872886657715),
                'NN' = c(0.20719945430755615, 0.1271975040435791, 0.23512053489685059, 0.19850337505340576, 0.198228120803833, 0.29349303245544434, 0.2691915035247803, 0.2730574607849121, 0.1871500015258789, 0.3756828308105469))

df_long = Excess_Risk %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")
df_long$Variable = factor(df_long$Variable, levels = c('PENN', 'NN'))

ggplot(df_long, aes(x=Variable, y=Value)) + geom_boxplot(fill="blue", alpha=0.2) + 
  xlab("") + ylab("Excess risk") + theme_bw()
