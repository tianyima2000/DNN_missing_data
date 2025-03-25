library(ggplot2)
library(tidyr)


##### Width of PI
PENN_MI = c(81.69155883789062, 81.78634643554688, 82.98922729492188, 82.98641204833984, 80.44266510009766, 84.17872619628906, 79.85308837890625, 79.39593505859375, 80.39218139648438, 83.87220001220703) 

NN_MI = c(84.18693542480469, 88.99688720703125, 86.26410675048828, 86.65713500976562, 85.01937866210938, 86.76568603515625, 86.96720123291016, 86.35366821289062, 85.93585205078125, 86.57105255126953) 

PENN_MF = c(80.99810028076172, 82.41928100585938, 82.72309112548828, 81.66561889648438, 80.74016571044922, 80.51212310791016, 81.15398406982422, 81.51428985595703, 82.26111602783203, 81.25188446044922) 

NN_MF = c(86.14031219482422, 87.12156677246094, 88.28395080566406, 86.90077209472656, 84.65249633789062, 87.34169006347656, 83.6952896118164, 84.34099578857422, 86.44054412841797, 86.71295928955078) 

PENN_II = c(83.04785919189453, 80.94291687011719, 83.23999786376953, 82.28594970703125, 80.26393127441406, 80.82980346679688, 81.31950378417969, 81.44747161865234, 81.7327880859375, 83.72972869873047) 

NN_II = c(86.64934539794922, 86.75347900390625, 86.78176879882812, 85.87229919433594, 82.97803497314453, 87.73680877685547, 86.1167984008789, 87.12030029296875, 86.08248901367188, 87.15596771240234) 

Excess_Risk = data.frame('PENN_MI' = PENN_MI, 'NN_MI' = NN_MI,
                         'PENN_MF' = PENN_MF, 'NN_MF' = NN_MF,
                         'PENN_II' = PENN_II, 'NN_II' = NN_II)
df_long = Excess_Risk %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")
df_long$Variable = factor(df_long$Variable, levels = c('PENN_MI', 'NN_MI', 'PENN_MF', 'NN_MF', 'PENN_II', 'NN_II'))

ggplot(df_long, aes(x=Variable, y=Value, fill=Variable)) + geom_boxplot(alpha=0.2) + 
  xlab("") + ylab("Width of PI") + theme_bw() + theme(legend.position = "none", text = element_text(size = 20)) + 
  scale_fill_manual(values = c('PENN_MI' = 'red', 'NN_MI' = 'blue', 'PENN_MF' = 'red', 
                               'NN_MF' = 'blue',  'PENN_II' = 'red', 'NN_II' = 'blue')) + scale_y_continuous(breaks = c(80, 82, 84, 86, 88))




##### Coverage
PENN_MI = c(0.9075, 0.9069, 0.9057, 0.9015, 0.8997, 0.9094, 0.8912, 0.8893, 0.9068, 0.9034) 

NN_MI = c(0.9062, 0.9047, 0.905, 0.9109, 0.8962, 0.9074, 0.8964, 0.8972, 0.9053, 0.8989) 

PENN_MF = c(0.9074, 0.9036, 0.9064, 0.9034, 0.8992, 0.9015, 0.8961, 0.8984, 0.9034, 0.9023) 

NN_MF = c(0.908, 0.9094, 0.9078, 0.9078, 0.8975, 0.9054, 0.8955, 0.8969, 0.9042, 0.8978) 

PENN_II = c(0.9037, 0.9008, 0.9066, 0.9051, 0.8958, 0.9107, 0.8972, 0.8927, 0.9072, 0.9014) 

NN_II = c(0.8968, 0.9061, 0.9052, 0.9084, 0.8944, 0.9099, 0.8917, 0.9012, 0.902, 0.8972) 

Excess_Risk = data.frame('PENN_MI' = PENN_MI, 'NN_MI' = NN_MI,
                         'PENN_MF' = PENN_MF, 'NN_MF' = NN_MF,
                         'PENN_II' = PENN_II, 'NN_II' = NN_II)
df_long = Excess_Risk %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")
df_long$Variable = factor(df_long$Variable, levels = c('PENN_MI', 'NN_MI', 'PENN_MF', 'NN_MF', 'PENN_II', 'NN_II'))

ggplot(df_long, aes(x=Variable, y=Value, fill=Variable)) + geom_boxplot(alpha=0.2) + 
  xlab("") + ylab("Coverage") + theme_bw() + theme(legend.position = "none", text = element_text(size = 20)) + 
  scale_fill_manual(values = c('PENN_MI' = 'red', 'NN_MI' = 'blue', 'PENN_MF' = 'red', 
                               'NN_MF' = 'blue',  'PENN_II' = 'red', 'NN_II' = 'blue')) 

