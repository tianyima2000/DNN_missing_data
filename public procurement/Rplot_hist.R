obs_prob = c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.00026717955704814365, 1.0, 1.0, 0.6023996616905212, 0.7893687307909399, 0.7889918483694085, 1.0, 0.3444156818475378, 1.0, 0.9984482352878727, 0.6269890279441507, 0.8150763585018835, 8.847005200269657e-05, 7.96230468024269e-05, 7.785364576237298e-05, 0.09706757165631862, 0.4124473824365714, 0.7155758604154907)
hist(obs_prob[obs_prob != 1])
sum(obs_prob != 1)

library(ggplot2)
library(dplyr)
data = data.frame(value = obs_prob[obs_prob != 1]) 

bin_edges = seq(0, 1, by = 0.1)

# Create a histogram
ggplot(data, aes(x = value)) +
  geom_histogram(breaks = bin_edges, fill = "steelblue", color = "white", alpha = 0.8) +
  labs(x = "Observational probability", y = "Frequency") + theme_bw() + theme(panel.grid.minor = element_blank(), text = element_text(size = 20)) +
  scale_x_continuous(breaks = bin_edges)
