# Hack so we can reload other modules without recompiling Julia

def import_julia_and_robot_dance():
	# To use PyJulia
	print('Loading Julia library...')
	from julia.api import Julia
	jl = Julia(compiled_modules=False)
	from julia import Main as Julia
	print('Loading Julia library... Ok!')
	print('Loading Robot-dance Julia module...')
	Julia.eval('include("robot_dance.jl")')
	print('Loading Robot-dance Julia module... Ok!')
