
pytorch_results <- c(0.814, 0.796, 0.81 , 0.812, 0.835, 0.823, 0.819, 0.799, 0.802, 0.802)
r_results <- c(0.834, 0.822, 0.838, 0.831, 0.857, 0.836, 0.839, 0.829, 0.826, 0.822)
baseline <- c(1.439, 1.469, 1.451, 1.469, 1.450, 1.443, 1.448, 1.447, 1.452, 1.443)

t.test(r_results - pytorch_results, conf.level = 0.99) # 1.873e-07
t.test(baseline - pytorch_results, conf.level = 0.99) # 1.767e-15
t.test(r_results - baseline, conf.level = 0.99) # 8.938e-16

rb <- t.test(baseline, conf.level = 0.99)
rb
rb$estimate - rb$conf.int

rr <- t.test(r_results, conf.level = 0.99)
rr
rr$estimate - rr$conf.int


rp <- t.test(pytorch_results, conf.level = 0.99)
rp
rp$estimate - rp$conf.int




