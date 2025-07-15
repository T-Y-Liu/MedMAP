        if self.is_training:
            w = F.softmax(self.weights, dim=0)

            p_mix = w[2] * t1_x1 + w[3] * t2_x1 + w[1] * t1ce_x1 + w[0] * flair_x1

            align_loss = (F.mse_loss(t1_x1, p_mix) +
                        F.mse_loss(t2_x1, p_mix) +
                        F.mse_loss(t1ce_x1, p_mix) +
                        F.mse_loss(flair_x1, p_mix)) / 8

            final_features = []
            modal_features = [flair_x1, t1ce_x1, t1_x1, t2_x1]
            for i in range(4):
                if mask[0, i].item():
                    final_features.append(modal_features[i])  
                else:
                    final_features.append(p_mix)  
           
            flair_x1, t1ce_x1, t1_x1, t2_x1 = final_features

        else:
            align_loss = torch.tensor(0.0, device=x.device)

            w = F.softmax(self.weights, dim=0)

            modal_features = [flair_x1, t1ce_x1, t1_x1, t2_x1]
            available_features = []
            available_weights = []
            for i in range(4):
                if mask[0, i].item():
                    available_features.append(modal_features[i])
                    available_weights.append(w[i])


            weights = torch.stack(available_weights)
            weights = weights / weights.sum()  
            p_mix = sum(wi * fi for wi, fi in zip(weights, available_features))

            final_features = []
            for i in range(4):
                if mask[0, i].item():
                    final_features.append(w[i] * modal_features[i])  
                else:
                    final_features.append(w[i] * p_mix)              

            flair_x1, t1ce_x1, t1_x1, t2_x1 = final_features