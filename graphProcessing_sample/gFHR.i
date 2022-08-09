# Channel Power Fractions
pow_1 =  1 #0.381
pow_2 =  1 #0.282
pow_3 =  1 #0.193
pow_4 =  1 #0.109
pow_5 =  1 #0.0347
#Channel Area fractions
frac_1 = 1
frac_2 = 1
frac_3 = 1
frac_4 = 1
frac_5 = 1
#Number of elements in the active core region
fuel_n = 3
expan_n = 3
n_core = 20
cont_n = 12
#Core geometry
fuel_dz = 0.3965
expan_dz = 0.3965
active_dz = 3.1
cont_dz = 1.2235
#Axial bottom
fuel_z0 = -5.94
expan_z0 = ${fparse fuel_z0 + fuel_dz}
active_z0 = ${fparse expan_z0 + expan_dz}
cont_z0 = ${fparse active_z0 + active_dz}

[EOS]
    [./eos] #Flibe
        type = SaltEquationOfState
    [../]
    [./eos2] #Solar salt
        type = SaltEquationOfState
        salt_type = Nitrate
    [../]
    [./air]
      type = AirEquationOfState
    [../]
    [./water]
      type = PTFunctionsEOS
      rho = water-rho
      k = water-k
      cp = water-cp
      mu = water-mu
      T_min = 273.15
      T_max = 373.15
    [../]
[]
[MaterialProperties]
    [./ss-mat]
        type = SolidMaterialProps
        k = 40
        Cp = 583.333
        rho = 6e3
    [../]

    [./graphite]
        type = SolidMaterialProps
        k = 30. # H451 data at high neutron fluence and elevated temperatures.
        Cp = 1665.5 # Butland and Maddison 1973/4 at 550C
        rho = 1740 #These values are based on KPs neutronics paper 12/2021
    [../]
    [./fuel-matrix]
        type = SolidMaterialProps
        k = 15. #These values are based on KPs neutronics paper 12/2021
        Cp = 1665.5 # Butland and Maddison 1973/4 at 550C
        rho = 1740 #These values are based on KPs neutronics paper 12/2021
    [../]
    [./pebble-core]
        type = SolidMaterialProps
        k = 15. #These values are based on KPs neutronics paper 12/2021
        Cp = 1665.5 # Butland and Maddison 1973/4 at 550C
        rho = 1740 #These values are based on KPs neutronics paper 12/2021
    [../]
    [./pebble-shell]
        type = SolidMaterialProps
        k = 15. #These values are based on KPs neutronics paper 12/2021
        Cp = 1665.5 # Butland and Maddison 1973/4 at 550C
        rho = 1740 #These values are based on KPs neutronics paper 12/2021
    [../]
[]
[Functions]
  [./power_profile]
    type = PiecewiseLinear
    x = '0            0.155       0.31        0.465       0.62        0.775       0.93        1.085       1.24        1.395       1.55        1.705       1.86        2.015       2.17        2.325       2.48        2.635       2.79        2.945 3.1'
    y = '0.36716	    0.33839	    0.35199	    0.37375	    0.38544	    0.39425	    0.39384	    0.39206	    0.38286	    0.37266	    0.3569	    0.34099	    0.32145	    0.30184	    0.27964	    0.25719	    0.23376	    0.20964	    0.19257	    0.20525  0.20525'
    scale_factor = 3.1
    axis = x
  [../]
 #####################################################2
    [./water-rho]
      type = PiecewiseLinear
      data_file = water_eos_P101325_1000.csv
      xy_in_file_only = false
      x_index_in_file = 0
      y_index_in_file = 1
      format = columns
    [../]
    [./water-cp]
      type = PiecewiseLinear
      data_file = water_eos_P101325_1000.csv
      xy_in_file_only = false
      x_index_in_file = 0
      y_index_in_file = 2
      format = columns
    [../]
    [./water-k]
      type = PiecewiseLinear
      data_file = water_eos_P101325_1000.csv
      xy_in_file_only = false
      x_index_in_file = 0
      y_index_in_file = 3
      format = columns
    [../]
    [./water-mu]
      type = PiecewiseLinear
      data_file = water_eos_P101325_1000.csv
      xy_in_file_only = false
      x_index_in_file = 0
      y_index_in_file = 4
      format = columns
    [../]
##################################################3
    [./rho_func]
        type = PiecewiseLinear
        x = '0'
        y = '0'
    [../]
    [./time_step]
        type = PiecewiseLinear
        x = '-1000 -980    1    10 100 1200 1240 1249 2500 2510	2600 3100'
        y = '1 5.0    5.0 5 5 5  5   1.   1    10	  100  100'
    [../]
    [./PumpHead]
        type = PiecewiseLinear
        x = '0'
        y = '1'
        scale_factor = 890484.0253
    [../]
    [./HX_v_in]
        type = PiecewiseLinear
        x = '1250'
        y = '1'
        scale_factor = 1.470135986
    [../]
    [./HX_T_in]
        type = PiecewiseLinear
        x = '1250'
        y = '702.1263979'
    [../]

[]

[ComponentInputParameters]
  [PB-Core-Channel]
    type = PBCoreChannelParameters
    Dh = 0.04
    D_heated = ${fparse 4/225}
    width_of_hs = '0.0138 0.0042 0.002'
    power_fraction = '0 0 0'
    n_heatstruct = 3
    elem_number_of_hs = '5 5 2'
    name_of_hs = 'pebble-core fuel-matrix pebble-shell'
    material_hs = 'pebble-core fuel-matrix pebble-shell'
    HT_surface_area_density = 225
    eos = eos
    orientation = '0 0 1 '
    dim_hs = 1
    roughness = 0.000015
    fuel_type = sphere
    porosity = 0.4
    d_pebble = 0.04
    HTC_geometry_type = PebbleBed
    WF_geometry_type = PebbleBed
    fluid_conduction = true
  []

######################################################2
    [./SC80-24in]
        type = PBOneDFluidComponentParameters #type = PBPipeParameters
        A = 0.47171642012 #2*0.23585821006
        Dh = 0.548
        material_wall = ss-mat
        wall_thickness = 0.031
        n_wall_elems = 3
        radius_i = 0.274
        eos = eos
        roughness = 0.000015
    [../]
    [./SC80-12in]
        type = PBOneDFluidComponentParameters #type = PBPipeParameters
        A = 0.13114 # assumes 2 loops
        Dh = 0.28894
        material_wall = ss-mat
        wall_thickness = 0.01748
        n_wall_elems = 3
        radius_i = 0.14447
        eos = eos
        roughness = 0.000015
    [../]
###################################################################3
    [./SC40-24in]
        type = PBOneDFluidComponentParameters #type = PBPipeParameters
        A = 1.0386891 #4*0.25967227
        Dh = 0.610
        material_wall = ss-mat
        wall_thickness = 0.0175
        n_wall_elems = 3
        radius_i = 0.2875
        eos = eos2
        roughness = 0.000015
    [../]

[]

[Components]
####################################################################
#####################  REACTOR CORE    #############################
####################################################################
    [./pke]
      type = PointKinetics
      lambda = '0.0133 0.0322 0.1197 0.3023 0.8530 2.8563'
      rho_fn_name = rho_func
      LAMBDA = 278E-6
      betai =  '1.81684E-4 1.03167E-3 9.29313E-4 2.04588E-3 8.83361E-4 3.52778E-4'
      feedback_components = 'active active-R'
      feedback_start_time = 1250
      irk_solver = true
    [../]
    [./reactor]
        type = ReactorPower
        initial_power = 320e6				# Initial total reactor power
        pke = 'pke'
        operating_power = 320e6
        enable_decay_heat = true
        isotope_fission_fraction = '1.0 0.0 0.0 0.0'
    [../]
    [./fueling]
        type = PBCoreChannel
        input_parameters = PB-Core-Channel
        position = '.0 6.74945 ${fuel_z0}'
        A = ${fparse 0.653251467*frac_4}
        length = ${fuel_dz}
        n_elems = ${fuel_n}
        #initial_V = 1.00811
       # initial_T = 823.15
        #initial_P = 593687.5628
       # Ts_init = 823.15
	   initial_V = 1.004697
	   initial_T = 822.698684
	   initial_P = 949107.946505
	   Ts_init = 822.699014
    [../]
    [./Branch1] # In to injection plenum
        type = PBSingleJunction
        inputs = 'fueling(out)'
        outputs = 'expansion(in)'
        eos = eos
        #initial_P = 1.8e5
		initial_V = 1.004697
		initial_T = 822.698803
		initial_P = 747071.300753
    [../]
    [./expansion]
        type = PBCoreChannel
        input_parameters = PB-Core-Channel
        position = '.0 6.84945 ${expan_z0}'
        A = ${fparse 1.183350407*frac_5}
        length = ${expan_dz}
        n_elems = ${expan_n}
        initial_V = 0.596815
        initial_T = 823.15
        initial_P = 580059.6807
        Ts_init = 823.15
    [../]
    [./Branch2] # In to injection plenum
        type = PBSingleJunction
        inputs = 'expansion(out)'
        outputs = 'active(in)'
        eos = eos
       # initial_P = 1.8e5
	   initial_V = 0.554629
       initial_T = 822.699170
       initial_P = 607672.977144
    [../]
    #########################################################2
    [./active]
      # These values are based on the KP neutronics paper and the fact that UM pcm/K values summed to the core average value.
      # We have assumed that all nodes care an equal weight.
      pke_material_type = 'Moderator Moderator None '
      n_layers_moderator = '${fparse n_core} ${fparse n_core}'
      moderator_reactivity_coefficients = '${fparse -2.0000E-08*40*5/n_core};
                                           ${fparse -2.2800E-07*40*5/n_core}' # Fuel region feedback mod formulation
      coolant_density_reactivity_feedback = true
      n_layers_coolant = ${fparse n_core}
      coolant_reactivity_coefficients = '-4.3836E-06'

      type = PBCoreChannel
      input_parameters = PB-Core-Channel
      position = '.0 6.84945 ${active_z0}'
      A = ${fparse 1.809557368*frac_5}
      length = ${active_dz}
      n_elems = ${fparse n_core}
      initial_V = 0.481173
      initial_T = 873.15
      initial_P = 548338.4491
      Ts_init = 950
      power_fraction = ' 0.0 ${fparse pow_5} 0.0 '
      power_shape_function = power_profile
    [../]
    [./active-R]
      moderator_reactivity_feedback = true
      n_layers_moderator = ${fparse n_core}
      moderator_reactivity_coefficients = ${fparse 0.92e-5/n_core}
      type = PBCoupledHeatStructure
      position = '0 6.44945 ${active_z0}'
      length =  ${active_dz}
      HS_BC_type = 'Coupled Coupled'
      name_comp_left = 'active'
      name_comp_right = 'down-active'
      HT_surface_area_density_left = 4.166666667
      HT_surface_area_density_right = 19.72899729
      material_hs = 'graphite ss-mat'
      width_of_hs = '0.6 0.02'
      elem_number_radial ='6 2'
      radius_i = 1.2
      T_bc_right = 823.15
      elem_number_axial = ${fparse n_core}
        # Ts_init = 900
    	Ts_init = 823.008181
    [../]
    [./Branch3] # In to injection plenum
        type = PBSingleJunction
        inputs = 'active(out)'
        outputs = 'contraction(in)'
        eos = eos
        #initial_P = 1.8e5
		initial_V = 0.371058
		initial_T = 923.264003
		initial_P = 297336.871112
    [../]
    [./contraction]
        type = PBCoreChannel
        input_parameters = PB-Core-Channel
        position = '.0 6.84945 ${cont_z0}'
        A = ${fparse 1.171185741*frac_5}
        length = ${cont_dz}
        n_elems = ${cont_n}
        #initial_V = 0.747062
        #initial_T = 923.15
        #initial_P = 517211.2449
        #Ts_init = 923.15
		initial_V = 0.573321
        initial_T = 923.641053
		initial_P = 234943.287298
		Ts_init = 923.967790
    [../]
    [./Branch030] #Outlet plenum
        type = PBVolumeBranch
        inputs = 'contraction(out)'
        outputs = 'pipe040(in)'
        center = '0 7.0744495 -0.79375'
        volume = 0.99970002
        K = '583.1 0.0006'			# This value was tuned to match the pressure drop in the 2-D core
        Area = 0.2524495
        eos = eos
        #initial_V = 2.673508074
        #initial_T = 923.1267449
        #initial_P = 165476
        width = 1.249999
        height = 0.0675
		initial_V = 2.659904
        initial_T = 923.957007
        initial_P = 165474.731744
    [../]
    [./Branch280] #Bottom branch
        type = PBVolumeBranch
        inputs = 'down-fuel(out)'
        outputs = 'fueling(in)'
        center = '0 7.3617 -6.04'
        volume = 0.2655022
        K = '0.35964 0.0'
        Area = 1.327511
        eos = eos
        #initial_P = 1.15402E+06
        #initial_V = 0.496077699
        #initial_T = 823.15
        width = 1.8245
        height = 0.2
		initial_V = 0.494399
        initial_T = 822.698563
        initial_P = 1153885.579124
    [../]
####################################################################

####################################################################
#####################  Reflector CORE    #############################
####################################################################

    # Core Reflector
    [./fueling-R]
      type = PBCoupledHeatStructure
      position = '0 6.44945 ${fuel_z0}'
      length =  ${fuel_dz}
      HS_BC_type = 'Coupled Coupled'
      name_comp_left = 'fueling'
      name_comp_right = 'down-fuel'
      HT_surface_area_density_left = 6.93481276
      HT_surface_area_density_right = 19.72899729
      material_hs = 'graphite ss-mat'
      width_of_hs = '1.079 0.02'
      elem_number_radial ='6 2'
      radius_i = 0.721
      T_bc_right = 823.15
      elem_number_axial = ${fuel_n}
      #Ts_init = 823.15
	  Ts_init = 823.182735
    [../]
    [./expansion-R]
      type = PBCoupledHeatStructure
      position = '0 6.44945 ${expan_z0}'
      length =  ${expan_dz}
      HS_BC_type = 'Coupled Coupled'
      name_comp_left = 'expansion'
      name_comp_right = 'down-expan'
      HT_surface_area_density_left = 5.15250301
      HT_surface_area_density_right = 19.72899729
      material_hs = 'graphite ss-mat'
      width_of_hs = '0.82959785 0.02'
      elem_number_radial ='6 2'
      radius_i = 0.97040215
      T_bc_right = 823.15
      elem_number_axial = ${expan_n}
      #Ts_init = 823.15
	  Ts_init = 814.142516
    [../]
    [./contract-R]
      type = PBCoupledHeatStructure
      position = '0 6.44945 ${cont_z0}'
      length =  ${cont_dz}
      HS_BC_type = 'Coupled Coupled'
      name_comp_left = 'contraction'
      name_comp_right = 'down-cont'
      HT_surface_area_density_left = 6.443059169
      HT_surface_area_density_right = 19.72899729
      material_hs = 'graphite ss-mat'
      width_of_hs = '0.599013771 0.02'
      elem_number_radial ='6 2'
      radius_i = 1.200986229
      T_bc_right = 823.15
      elem_number_axial = ${cont_n}
      #Ts_init = 873.15
	  Ts_init = 827.144314
    [../]
    [./top-inner-R]
      type = PBCoupledHeatStructure
      position = '0 6.44945 -0.76'
      length =  0.7325
      HS_BC_type = 'Adiabatic Coupled'
      name_comp_right = 'pipe040'
      HT_surface_area_density_right = 3.920174
      material_hs = 'graphite'
      width_of_hs = '1.249999'
      elem_number_radial ='6'
      radius_i = 0.0
      elem_number_axial = 7
      Ts_init = 923.15
    [../]
    [./top-outer-R]
      type = PBCoupledHeatStructure
      position = '0 6.44945 -0.76'
      length =  0.7325
      HS_BC_type = 'Coupled Coupled'
      name_comp_left = 'pipe040'
      name_comp_right = 'down-40'
      HT_surface_area_density_left = 3.920174
      HT_surface_area_density_right = 19.72899729
      material_hs = 'graphite ss-mat '
      width_of_hs = '0.550001 0.02'
      elem_number_radial ='6 2'
      radius_i = 1.249999
      T_bc_right = 823.15
      elem_number_axial = 7
      Ts_init = 873.15
    [../]


####################################################################
#####################      HOT LEG     #############################
####################################################################

    [./pipe040] # Lower Hot salt extraction pipe
        type = PBOneDFluidComponent
        input_parameters = SC80-24in
        position = '0 7.699449 -0.76'
        orientation = '0 0 1'
        length = 0.7325 #3.77
        n_elems = 7
        #initial_V = 1.43079
        #initial_T = 923.15
        #initial_P = 181852.7051
		initial_V = 1.296194
        initial_T = 918.564875
        initial_P = 163124.075322
    [../]

############################################################3
    [./Branch260] #Top branch
        type = PBVolumeBranch
        inputs = 'pipe040(out)'
        outputs = 'pipe050(in) hot_to_diode(in)'
        center = '0 7.699449 -0.0275'
        volume = 0.132104
        K = '0.00636 0.0 10.0'
        Area = 0.264208
        eos = eos
        #initial_T = 923.15
        #initial_P = 92669.9
        #initial_V = 2.554524377
        height = 0.1
		initial_V = 2.314221
		initial_T = 918.563905
		initial_P = 93794.905689
    [../]
    [./pipe050] #Reactor vessel to hot salt well
        type = PBOneDFluidComponent #type = PBPipe
        input_parameters = SC80-24in
        position = '0 7.949449 -0.0275'
        orientation = '0 2.244121 0.081'
        length = 2.245582344
        n_elems = 4
        #initial_V = 1.4310000
        #initial_P = 96222.4888889
        #initial_T = 924.0477778
		initial_V = 1.296192
        initial_T = 918.563951
        initial_P = 96572.364161
        Twall_init = 918.567066
    [../]
####################################################################

####################################################################
#####################     DOWNCOMER    #############################
####################################################################
    [./pipe130] #Stand pipe to reactor vessel
        type = PBOneDFluidComponent #type = PBPipe
        input_parameters = SC80-24in
        position = '0 14.7263 -0.1164969'
        orientation = '0 -6.45235 0.0889969'
        length = 6.45296373542
        n_elems = 9
        #initial_P = 978098.4737
        #initial_T = 823.1245789
        #initial_V = 1.39596
		initial_V = 1.261716
        initial_T = 808.534044
        initial_P = 994202.982252
        Twall_init = 808.535648
    [../]
    [./Branch607] # In to injection plenum
        type = PBBranch
        inputs = 'pipe130(out) cold_to_diode(in)'
        outputs = 'down-40(in)'
        eos = eos
        Area = 0.47171642012
        K = '0.0 0.0 0.0'
        #initial_P = 593481.3
		initial_V = 1.261716
        initial_T = 808.534101
        initial_P = 993650.370895
    [../]
    [./down-40]
      type = PBOneDFluidComponent
      eos = eos
      position = '0 8.27395 -0.0275'
      orientation = '0 0 -1'
      roughness = 0.000015
      A = 0.579623845
      Dh = 0.1
      length = 0.8 #3.77 # This is slighlty longer then pipe 40 due to the outlet plenum
      n_elems = 7
      #initial_V = 1.13208E+00
      #initial_P = 1.04465E+06
      #initial_T = 823.15
	  initial_V = 1.026822
	  initial_T = 808.521800
	  initial_P = 1060937.049476
    [../]
    [./down-cont]
      type = PBOneDFluidComponent
      eos = eos
      position = '0 8.27395 ${fparse cont_z0+cont_dz}'
      orientation = '0 0 -1'
      roughness = 0.000015
      A = 0.579623845
      Dh = 0.1
      length = ${cont_dz}
      n_elems = ${cont_n}
     # initial_V = 1.13208E+00
     # initial_P = 1.05844E+06
     # initial_T = 823.15
	 initial_V = 1.026816
	 initial_T = 808.497657
	 initial_P = 1074811.839232
    [../]
    [./down-active]
      type = PBOneDFluidComponent
      eos = eos
      position = '0 8.27395 ${cont_z0}'
      orientation = '0 0 -1'
      roughness = 0.000015
      A = 0.579623845
      Dh = 0.1
      length = ${active_dz}
      n_elems = ${fparse n_core}
      #initial_V = 1.13208E+00
      #initial_P = 1.09452E+06
      #initial_T = 823.15
	  initial_V = 1.026797
	  initial_T = 808.422603
	  initial_P = 1111105.121624
    [../]
    [./down-expan]
      type = PBOneDFluidComponent
      eos = eos
      position = '0 8.27395 ${active_z0}'
      orientation = '0 0 -1'
      roughness = 0.000015
      A = 0.579623845
      Dh = 0.1
      length = ${expan_dz}
      n_elems = ${expan_n}
      #initial_V = 1.13207E+00
      #initial_P = 1.13126E+06
      #initial_T = 823.15
	  initial_V = 1.026777
	  initial_T = 808.341480
	  initial_P = 1148071.463036
    [../]
    [./down-fuel]
      type = PBOneDFluidComponent
      eos = eos
      position = '0 8.27395 ${expan_z0}'
      orientation = '0 0 -1'
      roughness = 0.000015
      A = 0.579623845
      Dh = 0.1
      length = ${fuel_dz}
      n_elems = ${fuel_n}
      #initial_V = 1.13206E+00
      #initial_P = 1.14472E+06
      #initial_T = 823.15
	  initial_V = 1.026768
	  initial_T = 808.305840
	  initial_P = 1161610.570511
    [../]
    [./down-chain]
      type = PipeChain
      eos = eos
      component_names = 'down-40 down-cont down-active down-expan down-fuel'
      #junction_weak_constraint = true
    [../]

####################################################################

####################################################################
#####################     Reactor Vessel ###########################
####################################################################
    [./RV-40]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Coupled Adiabatic'
        position = '0 6.44945 -0.8275'
        orientation = '0 0 1'
        length = 0.8 #3.77 # This is slighlty longer then pipe 40 due to the outlet plenum
        elem_number_axial = 7
        material_hs = 'ss-mat'
        width_of_hs = '0.04'
        elem_number_radial = '5'
        radius_i = 1.87
        hs_type = cylinder
        #Ts_init = 823.15
        name_comp_left = 'down-40'
        HT_surface_area_density_left = 20.27100271
		Ts_init = 792.772238
    [../]
    [./RV-cont]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Coupled Adiabatic'
        position = '0 6.44945 ${cont_z0}'
        orientation = '0 0 1'
        length = ${cont_dz}
        elem_number_axial = ${cont_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.04'
        elem_number_radial = '5'
        radius_i = 1.87
        hs_type = cylinder
        #Ts_init = 823.15
        name_comp_left = 'down-cont'
        HT_surface_area_density_left = 20.27100271
		Ts_init = 792.748348
    [../]
    [./RV-active]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Coupled Adiabatic'
        position = '0 6.44945 ${active_z0}'
        orientation = '0 0 1'
        length = ${active_dz}
        elem_number_axial = ${fparse n_core}
        material_hs = 'ss-mat'
        width_of_hs = '0.04'
        elem_number_radial = '5'
        radius_i = 1.87
        hs_type = cylinder
       # Ts_init = 823.15
        name_comp_left = 'down-active'
        HT_surface_area_density_left = 20.27100271
		Ts_init = 792.674551
    [../]
    [./RV-expan]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Coupled Adiabatic'
        position = '0 6.44945 ${expan_z0}'
        orientation = '0 0 1'
        length = ${expan_dz}
        elem_number_axial = ${expan_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.04'
        elem_number_radial = '5'
        radius_i = 1.87
        hs_type = cylinder
        #Ts_init =823.15
        name_comp_left = 'down-expan'
        HT_surface_area_density_left = 20.27100271
		Ts_init = 792.594797
    [../]
    [./RV-fuel]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Coupled Adiabatic'
        position = '0 6.44945 ${fuel_z0}'
        orientation = '0 0 1'
        length = ${fuel_dz}
        elem_number_axial = ${fuel_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.04'
        elem_number_radial = '5'
        radius_i = 1.87
        hs_type = cylinder
        #Ts_init = 823.15
        name_comp_left = 'down-fuel'
        HT_surface_area_density_left = 20.27100271
		Ts_init = 792.559757
    [../]
####################################################################

####################################################################
#####################    Fluid DIODE   #############################
####################################################################
    [./hot_to_diode]
      type = PBOneDFluidComponent # type = PBPipe
      input_parameters = SC80-12in
      position = '0 7.949449 0.0'
      orientation = '0 0.1622505 -0.015 '
      length = 0.1629424
      n_elems = 2
      fluid_conduction = true
	  initial_V = 0.000002
      initial_T = 918.564875
      initial_P = 99282.398924
      Twall_init = 1298.316635
    [../]
    [./flow_diode]
      type = CheckValve
      open_area = 0.13114
      opening_pressure = 0.0
      inputs = 'hot_to_diode(out)'
      outputs = 'cold_to_diode(out)'
      eos = eos
	  initial_V = 0.000000
      initial_T = 822.927382
      initial_P = 978818.341798

    [../]
    [./cold_to_diode]
      type = PBOneDFluidComponent # type = PBPipe
      input_parameters = SC80-12in
      position = '0 8.27395 -0.03'
      orientation = '0 -0.1622505 0.015 '
      length = 0.1629424
      n_elems = 2
      fluid_conduction = true
	  initial_V = -0.000000
	  initial_T = 822.928841
	  initial_P = 995010.720152
	  Twall_init = 822.939897
    [../]
####################################################################4
    [./Branch501] #Primary tank branch
        type = PBVolumeBranch
        inputs = ' pipe050(out)'
        outputs = 'pipe060(in) pipe2(in)'
        center = '0 10.25 0.0525'
        volume = 0.0264208
        K = '0 0.3744 0.3744'
        K_reverse = '0.0 0.3744 0.3744'
        Area = 0.264208
        eos = eos
        #initial_P = 91003.7
        #initial_T = 923.15
        #initial_V = 0
        width = 0.1
		initial_V = 2.314222
		initial_T = 918.564003
		initial_P = 92146.865255
    [../]
    [./pipe060] #Hot well
        type = PBOneDFluidComponent # type = PBPipe
        input_parameters = SC80-24in
        position = '0 10.30 0.0525'
        orientation = '0 1.970888 0.34'
        length = 2.00
        n_elems = 4
        #initial_V = 1.43078
        #initial_P = 91320.7555556
       # initial_T = 923.15
	   initial_V = 1.296194
	   initial_T = 918.564043
	   initial_P = 91818.459788
	   Twall_init = 918.567158
    [../]
    [./Pump]
        type = PBPump
        inputs = 'pipe060(out)'
        outputs = 'pipe070(in)'
        eos = eos
        K = '0 0'
        K_reverse = '2000000 2000000'
        Area = 0.47171642012
        Head_fn = PumpHead
		initial_V = 1.296194
        initial_T = 918.564085
        initial_P = 1075555.219594
    [../]
    [./pipe070] #Hot salt well to CTAH
        type = PBOneDFluidComponent # type = PBPipe
        input_parameters = SC80-24in
        position = '0 12.271 0.3925'
        orientation = '0 3.22967 -0.046'
        length = 3.23
        n_elems = 6
        #initial_V = 1.43067
        #initial_T = 923.15
        #initial_P = 1.07540E+06
		initial_V = 1.296194
        initial_T = 918.564148
        initial_P = 1075920.802168
        Twall_init = 918.567262
    [../]
    [./Branch601] # In to hot manifold
        type = PBBranch
        inputs = 'pipe070(out)'
        outputs = 'HX(secondary_out)'
        eos = eos
        K = '0.16804 0.16804'
        Area = 0.3041
        #initial_T = 923.15
        #initial_P = 545696.49
        #initial_V = 2.219405893
		initial_V = 2.010641
        initial_T = 918.564213
        initial_P = 1073687.840225
    [../]

####################################################################

####################################################################
#####################  HEAT EXCHANGER  #############################
####################################################################

    [./HX]
        type = PBHeatExchanger
        A_secondary = 0.36447602
        A = 2
        Dh_secondary = 0.0196
        Dh = 0.090377046
        length_secondary = 18.47
        length = 3.418
        position = '0 15.50067 0.3465'
        HT_surface_area_density_secondary = 204.08
        HT_surface_area_density = 219.2041251
        #initial_T_secondary = 855.183
        #initial_P_secondary = 528866.4382
        #initial_V_secondary = 1.82544
        #initial_T = 873.15
        #initial_P = 976396
        #initial_V = 1.0
        #Twall_init = 873.15
        n_wall_elems = 3
        n_elems = 34
        material_wall = ss-mat
        wall_thickness = 0.000889
        eos_secondary = eos
        eos = eos2
        radius_i = 0.0098
        orientation_secondary = '0 18.151 -3.418 '
        orientation = '0 0 1'
        hs_type = cylinder
        HX_type = Countercurrent
        end_elems_refinement = 5
		initial_V = 1.482510
        initial_T = 725.819960
        initial_P = 999422.730969
        initial_V_secondary = 1.650515
        initial_T_secondary = 852.296579
        initial_P_secondary = 1064379.160071
        Twall_init = 788.636720
    [../]

####################################################################

####################################################################
#####################     COLD LEG     #############################
####################################################################

    [./Branch604] # In to pipe to drain tank
        type = PBBranch
        inputs = 'HX(secondary_in)'
        outputs = 'pipe110(in)'
        eos = eos
        K = '0.15422 0.15422'
        K_reverse = '2000000 2000000'
        Area = 0.1924226
        #initial_P = 500867.743
        #initial_T = 823.15
        #initial_V = 3.422407773
		initial_V = 3.093047
        initial_T = 808.533980
        initial_P = 1045748.894216
    [../]
    [./pipe110] #CTAH to drain tank
        type = PBOneDFluidComponent # type = PBPipe
        input_parameters = SC80-24in
        position = '0 18.2055 -3.0715'
        orientation = '0 -3.4791917 -0.075'
        length = 3.48
        n_elems = 7
        #initial_V = 1.39598
        #initial_T = 823.15
        #initial_P = 1.03796E+06
		initial_V = 1.261716
        initial_T = 808.533940
        initial_P = 1054202.338746
        Twall_init = 808.535506
    [../]
    [./Branch605] # In to stand pipe
        type = PBSingleJunction
        inputs = 'pipe110(out)'
        outputs = 'pipe120(in)'
        eos = eos
        #initial_P = 593481.3
		initial_V = 1.261716
        initial_T = 808.533942
        initial_P = 1054854.191795
    [../]
    [./pipe120] #Stand pipe
        type = PBOneDFluidComponent # type = PBPipe
        input_parameters = SC80-24in
        position = '0 14.7263 -3.1464969'
        orientation = '0 0 1'
        length = 3.03
        n_elems = 6 #37
        #initial_P = 1.00864E+06
        #initial_V = 1.39598
        #initial_T = 823.15
		initial_V = 1.261716
        initial_T = 808.533964
        initial_P = 1024804.892691
        Twall_init = 808.535563
    [../]
    [./Branch606] # In to pipe to reactor vessel
        type = PBSingleJunction
        inputs = 'pipe120(out)'
        outputs = 'pipe130(in)'
        eos = eos
       # initial_P = 593481.3
	   initial_V = 1.261716
       initial_T = 808.533990
       initial_P = 994755.593629

    [../]

##################################################################


####################################################################
#####################   PRIMARY TANK   #############################
####################################################################

    [./pipe2] #Pipe to primary tank
        type = PBOneDFluidComponent #type = PBPipe
        input_parameters = SC80-24in
        position = '0 10.25 0.1525'
        orientation = '0 0 1'
        length = 0.1
        n_elems = 1
        #initial_T = 923.15
        #initial_P = 93559.5
        #initial_V = 1e-5
		initial_V = -0.000002
        initial_T = 923.264334
        initial_P = 94521.246361
        Twall_init = 923.253158
    [../]
    [./pool2] #Primary Loop Expansion Tank
        type = PBLiquidVolume
        center = '0 10.25 0.7025'
        inputs = 'pipe2(out)'
        K = '100.0'
        K_reverse = '100'
        Area = 1
        volume = 0.934509044
        initial_level = 0.434509044
        initial_T = 923.15
        initial_P = 101917.3186
        initial_V = 4.44E-06
        display_pps = false
        covergas_component = 'cover_gas2'
        eos = eos
    [../]
    [./cover_gas2]
	    type = CoverGas
	    n_liquidvolume = 1
	    name_of_liquidvolume = 'pool2'
	    initial_P = 93559.58196
	    initial_Vol = 1.465490956
	    initial_T = 923.15
    [../]

####################################################################

####################################################################
#################### Intermediate Loop #############################
####################################################################

[./HX-in]
  type = PBTDJ
  input = 'HX(primary_in)'
  eos = eos2
  T_fn = HX_T_in
  v_fn = HX_v_in
  initial_V = 1.470192
  initial_T = 702.126873
  initial_P = 1030326.411376
[../]
[./HX-out]
  type = PBTDV
  input = 'HX(primary_out)'
  eos = eos2
  T_bc = 732.15
  p_bc = 968663.4908
  initial_V = 1.490705
  initial_T = 741.444607
  initial_P = 968663.400931
[../]
####################################################################5

####################################################################
#####################          RHT       ###########################
####################################################################

  [./RHT]
    type = SurfaceCoupling
    use_displaced_mesh = true
    coupling_type = RadiationHeatTransfer
    surface1_name = 'RV-40:outer_wall RV-cont:outer_wall RV-active:outer_wall RV-expan:outer_wall RV-fuel:outer_wall'
    surface2_name = 'RCCS-40:inner_wall RCCS-cont:inner_wall RCCS-active:inner_wall RCCS-expan:inner_wall RCCS-fuel:inner_wall'
    epsilon_1 = 0.8
    epsilon_2 = 0.8 # Based on a black coating
    radius_1 = 1.91
    use_nearest_node = true
    surface1_axis = 'z'
    surface2_axis = 'z'
  [../]

####################################################################

####################################################################
#####################     RCCS Wall      ###########################
####################################################################

    [./RCCS-40]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Adiabatic Coupled'
        position = '0 6.44945 -0.8275'
        orientation = '0 0 1'
        length = 0.8 #3.77 # This is slighlty longer then pipe 40 due to the outlet plenum
        elem_number_axial = 7
        material_hs = 'ss-mat'
        width_of_hs = '0.01'
        elem_number_radial = '5'
        radius_i = 2.24
        hs_type = cylinder
        #Ts_init = 336.15
        name_comp_right = 'riser-40'
        HT_surface_area_density_right = 19.7802
		Ts_init = 336.049106
    [../]
    [./RCCS-cont]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Adiabatic Coupled'
        position = '0 6.44945 ${cont_z0}'
        orientation = '0 0 1'
        length = ${cont_dz}
        elem_number_axial = ${cont_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.01'
        elem_number_radial = '5'
        radius_i = 2.24
        hs_type = cylinder
        #Ts_init = 336.15
        name_comp_right = 'riser-cont'
        HT_surface_area_density_right = 19.7802
		Ts_init = 335.819886
    [../]
    [./RCCS-active]
	    Ts_init = 335.221661
        type = PBCoupledHeatStructure
        HS_BC_type = 'Adiabatic Coupled'
        position = '0 6.44945 ${active_z0}'
        orientation = '0 0 1'
        length = ${active_dz}
        elem_number_axial = ${fparse n_core}
        material_hs = 'ss-mat'
        width_of_hs = '0.01'
        elem_number_radial = '5'
        radius_i = 2.24
        hs_type = cylinder
       # Ts_init = 336.15
        name_comp_right = 'riser-active'
        HT_surface_area_density_right = 19.7802
    [../]
    [./RCCS-expan]
	    Ts_init = 334.607001
        type = PBCoupledHeatStructure
        HS_BC_type = 'Adiabatic Coupled'
        position = '0 6.44945 ${expan_z0}'
        orientation = '0 0 1'
        length = ${expan_dz}
        elem_number_axial = ${expan_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.01'
        elem_number_radial = '5'
        radius_i = 2.24
        hs_type = cylinder
        #Ts_init = 336.15
        name_comp_right = 'riser-expan'
        HT_surface_area_density_right = 19.7802
    [../]
    [./RCCS-fuel]
        type = PBCoupledHeatStructure
        HS_BC_type = 'Adiabatic Coupled'
        position = '0 6.44945 ${fuel_z0}'
        orientation = '0 0 1'
        length = ${fuel_dz}
        elem_number_axial = ${fuel_n}
        material_hs = 'ss-mat'
        width_of_hs = '0.01'
        elem_number_radial = '5'
        radius_i = 2.24
        hs_type = cylinder
        #Ts_init = 336.15
		Ts_init = 334.383485
        name_comp_right = 'riser-fuel'
        HT_surface_area_density_right = 19.7802
    [../]
####################################################################

####################################################################
#####################     RCCS Hot       ###########################
####################################################################
    [./RCSS-HX-feed]
      type = PBOneDFluidComponent
      eos = water
      position = '0 8.69945 37.16'
      orientation = '0 1 0'
      length = 2.0
      n_elems = 4
      #initial_T = 326.5
      #initial_V = 2e-1
      #initial_P = 101325
      Dh = 0.434974
      A = 0.5943968
	  initial_V = 0.194588
	  initial_T = 323.777672
	  initial_P = 70935.569756
    [../]
    [./riser] # 4 18in schedule 40 pipes
      type = PBOneDFluidComponent
      eos = water
      position = '0 8.69945 -0.0275'
      orientation = '0 0 1'
      length = 37.1875
      n_elems = 20
      #initial_T = 326.5
      #initial_V = 2e-1
      3initial_P = 101325
      Dh = 0.434974
      A = 0.5943968
	  initial_V = 0.194588
	  initial_T = 323.651355
	  initial_P = 250945.142854
    [../]
    [./riser-40]
        type = PBOneDFluidComponent
        position = '0 8.69945 -0.8275'
        orientation = '0 0 1'
        length = 0.8 #3.77 # This is slighlty longer then pipe 40 due to the outlet plenum
        Dh = 0.05
        A = 0.7147
        eos = water
        n_elems = 7
        #initial_T = 326.5
        #initial_V = 1.7e-1
        #initial_P = 101325
        fluid_conduction = true
		initial_V = 0.161823
		initial_T = 323.414673
		initial_P = 434841.164483
    [../]
    [./riser-cont]
        type = PBOneDFluidComponent
        position = '0 8.69945 ${cont_z0}'
        orientation = '0 0 1'
        length = ${cont_dz}
        Dh = 0.05
        A = 0.7147
        eos = water
        n_elems = ${cont_n}
        #initial_T = 326
        #initial_V = 1.7e-1
        #initial_P = 101325
        fluid_conduction = true
		initial_V = 0.161804
		initial_T = 323.163481
        initial_P = 441737.741443
    [../]
    [./riser-active]
        type = PBOneDFluidComponent
        position = '0 8.69945 ${active_z0}'
        orientation = '0 0 1'
        length = ${active_dz}
        Dh = 0.05
        A = 0.7147
        eos = water
        n_elems = ${fparse n_core}
        #initial_T = 325
        #initial_V = 1.7e-1
        #initial_P = 101325
        fluid_conduction = true
        initial_V = 0.161757
        initial_T = 322.505446
        initial_P = 459781.806157
    [../]
    [./riser-expan]
        type = PBOneDFluidComponent
        position = '0 8.69945 ${expan_z0}'
        orientation = '0 0 1'
        length = ${expan_dz}
        Dh = 0.05
        A = 0.7147
        eos = water
        n_elems = ${expan_n}
        #initial_T = 324
        #initial_V = 1.7e-1
        #initial_P = 101325
        fluid_conduction = true
        initial_V = 0.161708
        initial_T = 321.834175
        initial_P = 478164.553752
    [../]
    [./riser-fuel]
        type = PBOneDFluidComponent
        position = '0 8.69945 ${fuel_z0}'
        orientation = '0 0 1'
        length = ${fuel_dz}
        Dh = 0.05
        A = 0.7147
        eos = water
        n_elems = ${fuel_n}
        #initial_T = 323
        #initial_V = 1.7e-1
        #initial_P = 101325
        fluid_conduction = true
        initial_V = 0.161691
        initial_T = 321.587927
        initial_P = 484898.842720
    [../]
    [./riser-chain]
      type = PipeChain
      eos = water
      component_names = 'riser-fuel riser-expan riser-active riser-cont riser-40 riser RCSS-HX-feed'
    [../]
####################################################################

####################################################################
#####################     RCCS HX       ###########################
####################################################################
    [./HX-J]
      type = PBSingleJunction
      outputs = 'RCSS-HX(secondary_in)'
      inputs = 'RCSS-HX-feed(out)'
      eos = water
	  initial_V = 0.194588
      initial_T = 323.783885
      initial_P = 70934.901491
    [../]
    [./RCSS-HX]
      type = PBHeatExchanger
      position = '0 10.69945 34.16'
      orientation = '0 0 1 '
      A_secondary  = 4.926189843
      A = 59.07381016
      Dh_secondary  = 0.0125222
      Dh = 0.106511061
      eos_secondary   = water
      eos = air
      HT_surface_area_density_secondary = 319.4326875
      HT_surface_area_density = 36.47139462
      wall_thickness = 0.0023114
      material_wall = ss-mat
      n_elems = 30
      n_wall_elems = 5
      length = 3
      #Twall_init = 325
      #initial_T_secondary = 325
      #initial_T = 307
      #initial_V = 2.5
      #initial_V_secondary = -2.5e-2
      hs_type = cylinder
      radius_i = 0.0062611
      initial_V = 2.343847
      initial_T = 306.700792
      initial_P = 101420.908275
      initial_V_secondary = -0.023467
      initial_T_secondary = 322.713980
      initial_P_secondary = 85474.070018
      Twall_init = 321.904807
    [../]
    [./chimney]
      type = PBOneDFluidComponent
      A = 64
      Dh = 4
      eos = air
      position = '0 10.69945 37.16'
      #initial_T = 311
      #initial_V = 2.3
      length = 7
      n_elems = 12
      #initial_T = 303.15
      initial_V = 2.186357
      initial_T = 309.949913
      initial_P = 101364.093049
    [../]
    [./chimnery-J]
      type = PBSingleJunction
      inputs = 'RCSS-HX(primary_out)'
      outputs = 'chimney(in)'
      eos = air
	  initial_V = 2.368678
      initial_T = 309.949965
      initial_P = 101402.713209
    [../]
    [./air-in]
      type = PBTDV
      eos = air
      p_bc = 101439.22
      T_bc = 303.15
      input = 'RCSS-HX(primary_in)'
    [../]
    [./air-out]
      type = PBTDV
      eos = air
      p_bc = 101325
      T_bc = 303.15
      input = 'chimney(out)'
    [../]
####################################################################

####################################################################
#####################     RCCS Cold Leg  ###########################
####################################################################
    [./Tank-branch]
      type = PBBranch
      inputs = 'RCSS-HX(secondary_out)'
      outputs = 'to-tank(in) RCSS-down(in)'
      K = '0.0 0.0 0.0'
      Area = 0.7147
      eos = water
	  initial_V = 0.161682
      initial_T = 321.464973
      initial_P = 99987.076025
    [../]
    [./to-tank]
      type = PBOneDFluidComponent
      Dh = 0.05
      A = 0.7147
      eos = water
      position = '0 10.69945 34.16'
      orientation = '0 1 0 '
      length = 1
      n_elems = 2
      #initial_V = 2.1e-6
      #initial_T = 320
	  initial_V = -0.000020
      initial_T = 320.112883
      initial_P = 99999.996583
    [../]
    [./RCSS-tank]
      #type = PBLiquidVolume
      type = PBTDV
      eos = water
      #center = '0 11.69945 34.16'
      #orientation = '0 0 1'
      #Area = 3.14
      #volume = 3.14
      #initial_level = 0.5
      #initial_T = 320
      T_bc = 320
      K = '100.'
      K_reverse = '100'
      inputs = 'to-tank(out)'
      input = 'to-tank(out)'
    [../]
    [./RCSS-down]
      type = PBOneDFluidComponent
      eos = water
      position = '0 10.69945 34.16'
      orientation = '0 0 -1'
      length = 40.1
      n_elems = 20
      #initial_T = 323
      #initial_P = 101325
      #initial_V = 1.7e-1
      Dh = 0.05
      A = 0.7147
	  initial_V = 0.161682
	  initial_T = 321.418040
	  initial_P = 294131.257782
    [../]
    [./RCCS-inlet]
      type = PBVolumeBranch
      volume = 1.0
      Area = 1.0
      eos = water
      inputs = 'RCSS-down(out)'
      outputs = 'riser-fuel(in)'
      K = '0.0 0.0'
      position = '0 0 0 '
      width = 2.0
      center = '0 9.69945 -6.19'
	  initial_V = 0.115555
      initial_T = 321.464384
      initial_P = 490704.810209
    [../]
#############

[]


[Postprocessors]
    [./core_in_T]
        type = ComponentBoundaryVariableValue
        input = 'fueling(out)'
        variable = temperature
    [../]
    [./core_out_T]
        type = ComponentBoundaryVariableValue
        input = 'pipe040(out)'
        variable = temperature
    [../]
    [./core_in_V]
        type = ComponentBoundaryVariableValue
        input = 'fueling(out)'
        variable = velocity
    [../]
    [./Core_energy]
        type = ComponentBoundaryEnergyBalance
        input = 'fueling(in) contraction(out)'
        eos = eos
    [../]
    [./core_out_v]
        type = ComponentBoundaryVariableValue
        input = 'pipe040(out)'
        variable = velocity
    [../]
    [./Core_flow]
        type = ComponentBoundaryFlow
        input = 'fueling(in)'
    [../]
    [./core_in_p]
        type = ComponentBoundaryVariableValue
        input = 'fueling(in)'
        variable = pressure
    [../]
    [./core_out_p]
        type = ComponentBoundaryVariableValue
        input = 'pipe040(out)'
        variable = pressure
    [../]
    [./DeltaP]
        type = DifferencePostprocessor
        value1 = core_in_p
        value2 = core_out_p
    [../]
    [./MaxFuel]
        type = ElementExtremeValue
        block = 'active:solid:fuel-matrix'
        variable = T_solid
        value_type = max
    [../]
    [./fueling-CVbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = fueling:pipe
    [../]
    [./fueling-CTbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = fueling:pipe
    [../]
    [./fueling-CPbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = fueling:pipe
    [../]
    [./fueling-CTsolidbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'fueling:solid:pebble-core fueling:solid:fuel-matrix fueling:solid:pebble-shell'
    [../]
    [./expansion-CVbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = expansion:pipe
    [../]
    [./expansion-CTbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = expansion:pipe
    [../]
    [./expansion-CPbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = expansion:pipe
    [../]
    [./expansion-CTsolidbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'expansion:solid:pebble-core expansion:solid:fuel-matrix expansion:solid:pebble-shell'
    [../]
    [./active-CVbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = active:pipe
    [../]
    [./active-CTbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = active:pipe
      outputs = csv
    [../]
    [./active-CPbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = active:pipe
      outputs = csv
    [../]
    [./active-CTsolidbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'active:solid:pebble-core active:solid:fuel-matrix active:solid:pebble-shell'
      outputs = csv
    [../]
    [./contraction-CVbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = contraction:pipe
      outputs = csv
    [../]
    [./contraction-CTbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = contraction:pipe
      outputs = csv
    [../]
    [./contraction-CPbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = contraction:pipe
      outputs = csv
    [../]
    [./contraction-CTsolidbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'contraction:solid:pebble-core contraction:solid:fuel-matrix contraction:solid:pebble-shell'
      outputs = csv
    [../]
    [./pipe040-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = pipe040
      outputs = csv
    [../]
    [./pipe040-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = pipe040
      outputs = csv
    [../]
    [./pipe040-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = pipe040
      outputs = csv
    [../]
    [./active-R-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'active-R:hs0 active-R:hs1'
      outputs = csv
    [../]
    [./contract-R-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'contract-R:hs0 contract-R:hs1'
      outputs = csv
    [../]
    [./expansion-R-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'expansion-R:hs0 expansion-R:hs1'
      outputs = csv
    [../]
    [./fueling-R-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'fueling-R:hs0 fueling-R:hs1'
      outputs = csv
    [../]

########################################################2
  [./external_Reactivity]
    type = FunctionValuePostprocessor
    function = rho_func
  [../]
  [./net_Reactivity]
    type = SumOfPostprocessors
    pps_names = 'external_Reactivity Total_Reactivity_Feedback'
  [../]
  [./doppler_Reactivity]
    type = SumOfPostprocessors
    pps_names = 'active:fuel-matrix_moderator_Reactivity'
  [../]
  [./moderator_Reactivity]
    type = SumOfPostprocessors
    pps_names = 'active:pebble-core_moderator_Reactivity'
  [../]
  [./coolant_Reactivity]
    type = SumOfPostprocessors
    pps_names = 'active:pipe_Coolant_Density_Reactivity'
  [../]
  [./reflector_Reactivity]
    type = SumOfPostprocessors
    pps_names = 'active-R_moderator_Reactivity'
  [../]

########################################################3
  [./run_time]
    type = PerfGraphData
    section_name = Root
    data_type = TOTAL
  [../]
  [./run_mem]
    type = PerfGraphData
    section_name = Root
    data_type = TOTAL_MEMORY
  [../]

########################################################4
    [./diode-flow]
        type = ComponentBoundaryFlow
        input = 'cold_to_diode(in)'
    [../]
    [./pipe050-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = pipe050
      outputs = csv
    [../]
    [./pipe050-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = pipe050
      outputs = csv
    [../]
    [./pipe050-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = pipe050
      outputs = csv
    [../]
#    [./pipe050-Tsolidbar]
#      type = AverageNodalVariableValue
#      variable = T_solid
#      block = 'pipe050:solid:wall_0'
#      outputs = csv
#    [../]
    [./pipe130-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = pipe130
      outputs = csv
    [../]
    [./pipe130-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = pipe130
      outputs = csv
    [../]
    [./pipe130-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = pipe130
      outputs = csv
    [../]
#    [./pipe130-Tsolidbar]
#      type = AverageNodalVariableValue
#      variable = T_solid
#      block = 'pipe130:solid:wall_0'
#      outputs = csv
#    [../]
    [./hot_to_diode-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = hot_to_diode
      outputs = csv
    [../]
    [./hot_to_diode-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = hot_to_diode
      outputs = csv
    [../]
    [./hot_to_diode-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = hot_to_diode
      outputs = csv
    [../]
#    [./hot_to_diode-Tsolidbar]
#      type = AverageNodalVariableValue
#      variable = T_solid
#      block = 'hot_to_diode:solid:wall_0'
#      outputs = csv
#    [../]
    [./cold_to_diode-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = cold_to_diode
      outputs = csv
    [../]
    [./cold_to_diode-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = cold_to_diode
      outputs = csv
    [../]
    [./cold_to_diode-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = cold_to_diode
      outputs = csv
    [../]
#    [./cold_to_diode-Tsolidbar]
#      type = AverageNodalVariableValue
#      variable = T_solid
#      block = 'cold_to_diode:solid:wall_0'
#      outputs = csv
#    [../]
    [./down-40-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = down-40
      outputs = csv
    [../]
    [./down-40-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = down-40
      outputs = csv
    [../]
    [./down-40-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = down-40
      outputs = csv
    [../]
    [./down-cont-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = down-cont
      outputs = csv
    [../]
    [./down-cont-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = down-cont
      outputs = csv
    [../]
    [./down-cont-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = down-cont
      outputs = csv
    [../]
    [./down-active-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = down-active
      outputs = csv
    [../]
    [./down-active-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = down-active
      outputs = csv
    [../]
    [./down-active-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = down-active
      outputs = csv
    [../]
    [./down-expan-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = down-expan
      outputs = csv
    [../]
    [./down-expan-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = down-expan
      outputs = csv
    [../]
    [./down-expan-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = down-expan
      outputs = csv
    [../]
    [./down-fuel-Vbar]
      type = AverageNodalVariableValue
      variable = velocity
      block = down-fuel
      outputs = csv
    [../]
    [./down-fuel-Tbar]
      type = AverageNodalVariableValue
      variable = temperature
      block = down-fuel
      outputs = csv
    [../]
    [./down-fuel-Pbar]
      type = AverageNodalVariableValue
      variable = pressure
      block = down-fuel
      outputs = csv
    [../]
    [./RV-40-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'RV-40:hs0'
      outputs = csv
    [../]
    [./RV-cont-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'RV-cont:hs0'
      outputs = csv
    [../]
    [./RV-active-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'RV-active:hs0'
      outputs = csv
    [../]
    [./RV-expan-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'RV-expan:hs0'
      outputs = csv
    [../]
    [./RV-fuel-HSbar]
      type = AverageNodalVariableValue
      variable = T_solid
      block = 'RV-fuel:hs0'
      outputs = csv
    [../]

#################################################################5
      [./HX_energy]
          type = HeatExchangerHeatRemovalRate
          heated_perimeter = ${fparse 204.08*0.36447602}
          block = 'HX:secondary_pipe'
      [../]
      [./pressure_outlet]
          type = ComponentBoundaryFlow
          input = 'pipe2(out)'
      [../]
      [./secondary-flow]
          type = ComponentBoundaryFlow
          input = 'HX(primary_out)'
      [../]
      [./IHX-in]
          type = ComponentBoundaryVariableValue
          input = 'HX(secondary_out)'
          variable = temperature
      [../]
      [./IHX-out]
          type = ComponentBoundaryVariableValue
          input = 'HX(secondary_in)'
          variable = temperature
      [../]
      [./pipe060-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = pipe060
        outputs = csv
      [../]
      [./pipe060-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = pipe060
        outputs = csv
      [../]
      [./pipe060-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = pipe060
        outputs = csv
      [../]
#      [./pipe060-Tsolidbar]
#        type = AverageNodalVariableValue
#        variable = T_solid
#        block = 'pipe060:solid:wall_0'
#        outputs = csv
#      [../]
      [./pipe070-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = pipe070
        outputs = csv
      [../]
      [./pipe070-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = pipe070
        outputs = csv
      [../]
      [./pipe070-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = pipe070
        outputs = csv
      [../]
#      [./pipe070-Tsolidbar]
#        type = AverageNodalVariableValue
#        variable = T_solid
#        block = 'pipe070:solid:wall_0'
#        outputs = csv
#      [../]
      [./pipe110-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = pipe110
        outputs = csv
      [../]
      [./pipe110-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = pipe110
        outputs = csv
      [../]
      [./pipe110-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = pipe110
        outputs = csv
      [../]
#      [./pipe110-Tsolidbar]
#        type = AverageNodalVariableValue
#        variable = T_solid
#        block = 'pipe110:solid:wall_0'
#        outputs = csv
#      [../]
      [./pipe120-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = pipe120
        outputs = csv
      [../]
      [./pipe120-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = pipe120
        outputs = csv
      [../]
      [./pipe120-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = pipe120
        outputs = csv
      [../]
#      [./pipe120-Tsolidbar]
#        type = AverageNodalVariableValue
#        variable = T_solid
#        block = 'pipe120:solid:wall_0'
#        outputs = csv
#      [../]
      [./pipe2-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = pipe2
        outputs = csv
      [../]
      [./pipe2-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = pipe2
        outputs = csv
      [../]
      [./pipe2-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = pipe2
        outputs = csv
      [../]
#      [./pipe2-Tsolidbar]
#        type = AverageNodalVariableValue
#        variable = T_solid
#        block = 'pipe2:solid:wall_0'
#        outputs = csv
#      [../]
      [./HX-PVbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = HX:primary_pipe
        outputs = csv
      [../]
      [./HX-PTbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = HX:primary_pipe
        outputs = csv
      [../]
      [./HX-PPbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = HX:primary_pipe
        outputs = csv
      [../]
      [./HX-SVbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = HX:secondary_pipe
        outputs = csv
      [../]
      [./HX-STbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = HX:secondary_pipe
        outputs = csv
      [../]
      [./HX-SPbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = HX:secondary_pipe
        outputs = csv
      [../]
      [./HX-PTsolidbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = HX:solid:wall
        outputs = csv
      [../]
#############################################################################6
      [./RCCS-EnergyBalance]
          type = ComponentBoundaryEnergyBalance
          input = 'riser-fuel(in) riser-40(out)'
          eos = water
      [../]
      [./RV-EnergyBalance]
          type = ConductionHeatRemovalRate
          heated_perimeter = 12.000884
          boundary = 'RV-40:outer_wall RV-cont:outer_wall RV-active:outer_wall RV-expan:outer_wall RV-fuel:outer_wall'
      [../]
      [./RV-surface]
          type = SideAverageValue
          variable = T_solid
          boundary = 'RV-40:outer_wall RV-cont:outer_wall RV-active:outer_wall RV-expan:outer_wall RV-fuel:outer_wall'
      [../]
      [./RCCS-InletTemp]
          type = ComponentBoundaryVariableValue
          input = 'riser-fuel(in)'
          variable = temperature
      [../]
      [./RCCS-OutletTemp]
          type = ComponentBoundaryVariableValue
          input = 'riser-40(out)'
          variable = temperature
      [../]
      [./air_T_in]
          type = ComponentBoundaryVariableValue
          variable = temperature
          input = 'RCSS-HX(primary_in)'
      [../]
      [./air_T_out]
          type = ComponentBoundaryVariableValue
          variable = temperature
          input = 'RCSS-HX(primary_out)'
      [../]
      [./air_flow]
          type = ComponentBoundaryFlow
          input = 'RCSS-HX(primary_in)'
      [../]
      [./water_flow]
          type = ComponentBoundaryFlow
          input = 'RCSS-HX(secondary_in)'
      [../]
      [./AIR-pickup]
        type = ComponentBoundaryEnergyBalance
        input = 'RCSS-HX(primary_in) RCSS-HX(primary_out)'
        eos = air
      [../]
      [./riser-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser
        outputs = csv
      [../]
      [./riser-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser
        outputs = csv
      [../]
      [./riser-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser
        outputs = csv
      [../]
      [./riser-40-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser-40
        outputs = csv
      [../]
      [./riser-40-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser-40
        outputs = csv
      [../]
      [./riser-40-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser-40
        outputs = csv
      [../]
      [./riser-cont-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser-cont
        outputs = csv
      [../]
      [./riser-cont-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser-cont
        outputs = csv
      [../]
      [./riser-cont-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser-cont
        outputs = csv
      [../]
      [./riser-active-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser-active
        outputs = csv
      [../]
      [./riser-active-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser-active
        outputs = csv
      [../]
      [./riser-active-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser-active
        outputs = csv
      [../]
      [./riser-expan-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser-expan
        outputs = csv
      [../]
      [./riser-expan-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser-expan
        outputs = csv
      [../]
      [./riser-expan-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser-expan
        outputs = csv
      [../]
      [./riser-fuel-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = riser-fuel
        outputs = csv
      [../]
      [./riser-fuel-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = riser-fuel
        outputs = csv
      [../]
      [./riser-fuel-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = riser-fuel
        outputs = csv
      [../]
      [./RCCS-40-HSbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = 'RCCS-40:hs0'
        outputs = csv
      [../]
      [./RCCS-cont-HSbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = 'RCCS-cont:hs0'
        outputs = csv
      [../]
      [./RCCS-active-HSbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = 'RCCS-active:hs0'
        outputs = csv
      [../]
      [./RCCS-expan-HSbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = 'RCCS-expan:hs0'
        outputs = csv
      [../]
      [./RCCS-fuel-HSbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = 'RCCS-fuel:hs0'
        outputs = csv
      [../]
      [./chimney-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = chimney
        outputs = csv
      [../]
      [./chimney-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = chimney
        outputs = csv
      [../]
      [./chimney-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = chimney
        outputs = csv
      [../]
      [./to-tank-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = to-tank
        outputs = csv
      [../]
      [./to-tank-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = to-tank
        outputs = csv
      [../]
      [./to-tank-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = to-tank
        outputs = csv
      [../]
      [./RCSS-down-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = RCSS-down
        outputs = csv
      [../]
      [./RCSS-down-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = RCSS-down
        outputs = csv
      [../]
      [./RCSS-down-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = RCSS-down
        outputs = csv
      [../]
      [./RCSS-HX-feed-Vbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = RCSS-HX-feed
        outputs = csv
      [../]
      [./RCSS-HX-feed-Tbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = RCSS-HX-feed
        outputs = csv
      [../]
      [./RCSS-HX-feed-Pbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = RCSS-HX-feed
        outputs = csv
      [../]
      [./RCSS-HX-PVbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = RCSS-HX:primary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-PTbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = RCSS-HX:primary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-PPbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = RCSS-HX:primary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-SVbar]
        type = AverageNodalVariableValue
        variable = velocity
        block = RCSS-HX:secondary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-STbar]
        type = AverageNodalVariableValue
        variable = temperature
        block = RCSS-HX:secondary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-SPbar]
        type = AverageNodalVariableValue
        variable = pressure
        block = RCSS-HX:secondary_pipe
        outputs = csv
      [../]
      [./RCSS-HX-PTsolidbar]
        type = AverageNodalVariableValue
        variable = T_solid
        block = RCSS-HX:solid:wall
        outputs = csv
      [../]


[]
#[VectorPostprocessors] # These VPps can be used to access the temperatures in the coolant and the fuel
#  [./coolant]
#    type = NodalValueSampler
#    block = 'active:pipe' # This name corresponds to the one in paraview
#    variable = temperature
#    sort_by = z
#    outputs = coolant_out # This is used to control the location of where they print
#    use_displaced_mesh = true # This is needed because of how SAM handles the mesh
#  [../]
#  [./fuel_low]
#    type = NodalValueSampler
#    block = 'active:solid:low'
#    variable = T_solid # For solids you are interested in T_solid not temperature
#    sort_by = z
#    outputs = fuel_out
#    use_displaced_mesh = true
#  [../]
#  [./fuel_kernal]
#    type = NodalValueSampler
#    block = 'active:solid:kernal'
#    variable = T_solid
#    sort_by = z
#    outputs = fuel_out
#    use_displaced_mesh = true
#  [../]
#  [./fuel_outer]
#    type = NodalValueSampler
#    block = 'active:solid:outer'
#    variable = T_solid
#    sort_by = z
#    outputs = fuel_out
#    use_displaced_mesh = true
#  [../]
#[]

#####################################################################################reactivity-avg-uniform#####################################





##########################################################################################global########3
[GlobalParams]
    global_init_P = 106349
    global_init_V = 1.814910e+00
    global_init_T = 923.15
    #Tsolid_sf = 1e-6
    display_pps = false
    #scaling_factor_var = '1 1 1e-6 '
    [./PBModelParams]
        p_order = 1
        low_advection_limit = 1e-9
        scaling_velocity = 0.00001
        fluid_conduction = true
    [../]
[]
[Preconditioning]
    [./SMP_PJFNK]
        type = SMP
        full = true
        solve_type = 'PJFNK'
        petsc_options_iname = '-pc_type -ksp_gmres_restart'
        petsc_options_value = 'lu 101'
    [../]
[]


[Outputs]
    print_linear_residuals = false
    perf_graph = true
    [./out]
        type = Checkpoint
        execute_on = 'initial timestep_end FAILED FINAL'
        interval = 5
    [../]
    [./out_displaced]
        type = Exodus
        use_displaced = true
        execute_on = 'initial FAILED timestep_end'
        interval = 30
        sequence = false
        output_material_properties = true
    [../]
    [./csv]
        type = CSV
        interval = 5
    [../]
    [./console]
        type = Console
        fit_mode = AUTO
        execute_scalars_on = 'NONE'
        perf_log = true
        output_nonlinear = false
        output_linear = false
        interval = 30
    [../]
  file_base = PB-FHR-multi-ss

[]
[Debug]
  #show_var_residual_norms = true
  #show_material_props = true
[]
#####################################################3reactor##################33###################################


[Executioner]
    start_time                     = -1000.0
    end_time                       = 1250
    [./TimeStepper]
        type = FunctionDT
        function = time_step
        dt = 1
    [../]

    type                           = Transient
    dt                             = 0.5
    dtmin                          = 1.e-5
    nl_rel_tol                     = 1e-6
    nl_abs_tol                     = 1e-3
    nl_max_its                     = 12
    l_max_its                      = 50
    #automatic_scaling = true
    [./Quadrature]
        type = TRAP
        order = FIRST
    [../]


[]
