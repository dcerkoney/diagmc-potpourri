all:
	make chi_ch_example
	make self_en_example

chi_ch_example:
	make -C build -f makefile.chi_ch
	ln -sfn build/hub_2dsqlat_rt_mcmc_chi_ch_example.exe hub_2dsqlat_rt_mcmc_chi_ch_example.exe

self_en_example:
	make -C build -f makefile.self_en
	ln -sfn build/hub_2dsqlat_rt_mcmc_self_en_example.exe hub_2dsqlat_rt_mcmc_self_en_example.exe

clean:
	make clean -C build -f makefile.chi_ch
	rm hub_2dsqlat_rt_mcmc_chi_ch_example.exe
	
	make clean -C build -f makefile.self_en
	rm hub_2dsqlat_rt_mcmc_self_en_example.exe
