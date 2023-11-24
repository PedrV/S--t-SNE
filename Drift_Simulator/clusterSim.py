import numpy as np


class ClusterSim(object):
	def __init__(self, name, initial_mean, initial_cov, final_mean, final_cov, translate_time, cooldown):
		self.name = name
		self.sims = -1
		self.ticks = 0

		self.translate_time = translate_time
		self.current_translate_time = -1
		self.cooldown = cooldown
		self.current_cooldown = cooldown
		
		self.mean = np.array(initial_mean).astype("float")
		self.cov = np.array(initial_cov).astype("float")

		self.means = [np.array(final_mean), np.array(initial_mean)]
		self.covs = [np.array(final_cov), np.array(initial_cov)]
		self.meanVariationPerTick = ((self.means[0] - self.mean)/self.translate_time)
		self.covVariationPerTcik = ((self.covs[0] - self.cov)/self.translate_time)

	def tick(self):
		self.ticks += 1


		if self.current_translate_time > 0:
			self.mean += self.meanVariationPerTick
			self.cov += self.covVariationPerTcik
			self.current_translate_time -= 1

		elif self.current_translate_time == 0:
			self.means.append(self.means.pop(0))
			self.covs.append(self.covs.pop(0))
			self.current_cooldown = self.cooldown
			self.current_translate_time = -1

		elif self.current_cooldown == 0:
			self.current_translate_time = self.translate_time

		else:
			self.meanVariationPerTick = ((self.means[0] - self.mean)/self.translate_time)
			self.covVariationPerTcik = ((self.covs[0] - self.cov)/self.translate_time)
			self.current_cooldown -= 1
		pass


	def get_n_points(self,n,numpy=False):
		points = []
		pointsRaw = np.random.multivariate_normal(size=n , mean=self.mean, cov=self.cov)
		if not numpy:
			for pointRaw in pointsRaw:
				self.sims += 1
				point = {f"{self.name}_{self.sims}_{i}":val for i,val in enumerate(pointRaw)}
				points.append(point)

		else:
			points = pointsRaw
			self.sims += len(pointsRaw)
		return points

		







