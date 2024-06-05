 import pyvista as pv

supertoroid = pv.ParametricSuperEllipsoid(exponent=2.5, aspect_ratio=(1, 1, 1))
ellipsoid = pv.ParametricEllipsoid()
partial_ellipsoid = pv.ParametricEllipsoid(start_phi=0, end_phi=pv.constants.pi / 2, start_theta=0, end_theta=pv.constants.pi / 2)
pseudosphere = pv.ParametricPseudosphere()
bohemian_dome = pv.ParametricBohemianDome()
bour = pv.ParametricBour()
boys_surface = pv.ParametricBoy()
catalan_minimal = pv.ParametricCatalanMinimal()
conic_spiral = pv.ParametricConicSpiral()
cross_cap = pv.ParametricCrossCap()
dini = pv.ParametricDini()
enneper = pv.ParametricEnneper(plot_position="yz")
figure_8_klein = pv.ParametricFigure8Klein()
henneberg = pv.ParametricHenneberg()
klein = pv.ParametricKlein()
kuen = pv.ParametricKuen()
mobius = pv.ParametricMobius()
plucker_conoid = pv.ParametricPluckerConoid()
random_hills = pv.RandomHills()
roman = pv.ParametricRoman()
super_ellipsoid = pv.ParametricSuperEllipsoid(exponent=1.618)
torus = pv.ParametricTorus()

circular_arc = pv.CircularArc(point1=(0, 0, 0), point2=(1, 0, 0), center=(0.5, 0, 0))
extruded_half_arc = circular_arc.extrude(height=1)
extruded_half_arc["show_edges"] = True

plotter = pv.Plotter()

plotter.add_mesh(supertoroid, color="lightblue")
plotter.add_mesh(ellipsoid, color="lightblue")
plotter.add_mesh(partial_ellipsoid, color="lightblue", plot_direction="z")
plotter.add_mesh(pseudosphere, color="lightblue")
plotter.add_mesh(bohemian_dome, color="lightblue")
plotter.add_mesh(bour, color="lightblue")
plotter.add_mesh(boys_surface, color="lightblue")
plotter.add_mesh(catalan_minimal, color="lightblue")
plotter.add_mesh(conic_spiral, color="lightblue")
plotter.add_mesh(cross_cap, color="lightblue")
plotter.add_mesh(dini, color="lightblue")
plotter.add_mesh(enneper, color="lightblue")
plotter.add_mesh(figure_8_klein, color="lightblue")
plotter.add_mesh(henneberg, color="lightblue")
plotter.add_mesh(klein, color="lightblue")
plotter.add_mesh(kuen, color="lightblue")
plotter.add_mesh(mobius, color="lightblue")
plotter.add_mesh(plucker_conoid, color="lightblue")
plotter.add_mesh(random_hills, color="lightblue")
plotter.add_mesh(roman, color="lightblue")
plotter.add_mesh(super_ellipsoid, color="lightblue")
plotter.add_mesh(torus, color="lightblue")
plotter.add_mesh(circular_arc, color="lightblue")
plotter.add_mesh(extruded_half_arc, color="lightblue")

plotter.show()