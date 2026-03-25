## Project Meeting Notes

constant e and b fields
solve newtons 2nd law
make equation dimensionless
e0 b0
v and sp dep func
ode pde solver function python

online ode solvers? python better

can add code to appendix reference figure important
can mention all in one chapter in one go

Time dep last extension maybe
Include textbook examples?




mirror point parallel v 0 

bounce times - dipole fields

test curveature and gradiaent b drift

orbit code
simple fields
think will get drift
work them out analytucally compare what get

does full orbit code does the guiding centre move in same way

let particles go through waves or something more complicated
step wise
test on simple fields
does it do e cross b

do want full orbit code NII L force
or want guiding centre code too

test push in z direction

e perpendic to b 
comprare guiding centre full particle orbit

check does grad b still give constant vector
v but be linear function
bz varies linearly in one direction
b0 x/l
mod B careful how you calculate gradient linear
careful
may not work as b=0

maube b0 + exp function. not sure about how that would work

charge direction for grad b

make b field more complicated
- time depend
- em wave oscillation
- include e field for faradays law

test constant field cases
plot anayltical solution from book and check 

plotting velocity and position 
take difference between two solutions

numerical solution calculated at discrete points in time 
at the time points where exist

one way to check take the difference
modulus at these ponts and see how big they are plot agaisnt time

much smaller than 1 should be

can check direction for charge
velocity field 

make sure methods work
go step by step
to make it more complicated


calculate particle in dipole fields
look at drifts around the dipole axis 
for bounce times expansion

if dipole has rotation axis and diploe not alligned. create field aceelerated particles
3d complicated 

analyse planets on orbit fields irregular on side
few space missions past outer planets
unusual patricle energies
relativistic
harder but possible

choose 1 or 2 applicartions 
legnth 20-25 pages
figures can be longer

shouldn’t be overly long and need to explain theory 
appendix with code used
and what have done

numerical options
test cases
done this and add figure code produces
use it to check drifts and what see
if code works should be same as guidung centre theory 

final more complicated 
apply this to dipole field
or if had rotating dipole field

more complicated application so many IC
don’t do too many
make sure coming to conclusion

check if guiding centre theory is good approx
may be good or bad. need to eval
if goes to 0 maybe its not 

nul points
might want ot explore this

take simple one and what happens if add particles 
when does it break down


first steps go on test code simple cases
3d with pushing up in z
add const e field
see if e cross b drift comes next

this is about investigating particle orbits
don’t need to reinvent
look up method
slove_ivp
try
RK4 maybe
has other things it can do 
look into when gets more complicated




dipole field in cartesian, avoid singularity
dimensionless 
natural length scale 
earth radius
electron/proton 
charge to mass ratio

functions e and b
inside vary
replace lines

inhomogenous field

dipole field and curvature and grad b field no e field
e field t dep

no rotation yet
static dipole

spherical?

dipole field in cartesian coord
z axis with symm axis with dipole

x =r sin
use trig
x

project b spehr
onto xyz

br blambda
into xyz
coord trans

also er elamda in ex ey ez

combines comp

can simplify final result?

spherical to xyz component sb
comp b in to b xyz




23/2

electromag book with given dipole field
called jackson electrodynamics?

compare dipole field to guiding centre
generic initial conditions
normalise
use earth scale with earth radius?
fig 3.2 from textbook, reproduce?
can use electron rest mass 511kev
1/2m_ev^2 = E0
me c^2. = 511kev
1/2 v^2/c^2 /L = Eo keV/511
check pitch angle
reach upper atmosphere? then what happens
pure dipole field must mirror due to inifinity?

IC for orbit
equtorial plane first so z=0
choose x or y
check that doesn’t matter for test
inital velocity 
want to go up or down first?
v parallel +ve/0-
pick x vy yz 
general behaviour
check discrepicy, refine guiding centre is inital spot but real particle sitting away from that?
or sitting on rim of gryo radius?
how far is that
can i put there without loss of generality
where is gc when start there. reverse test
guiding centre eqns and compare the two seperate code later?

all drifts need to compare
all equations of motion dipole field not just lorentz force eqn
full eqn montion
3 eqns for gc position?
1 extra for v_parallel
v perp also
calculate deriv of E and B
curvature 
RHS of diff eq is more complicated
hidden in notation and alot more algebra
maple, mathimaica or matlab come may be needed if more complicated and import into code to do derivative
there may be multiple drifts need to check

How to start:
try orbit in dipole
radius z=0 plane in x and y
deicde if want particle to go up +ve v parall +ve
deicde how to split vparallel and v perpendicular
total energy pitch angle
position
from there calc vx vy vz
if on x axis v perp in y direction = vy
from v^2
done.
vx =0 vy, vz then test
pay attention to step size
energy dep may take a while to bounce

make sure my tools do what want them to do to show results

Presentation
Can share slides in advance
can show code maybe
one slide of eqns max probably
explain conecpts and link to equations
present to the students
dipole field example if it works
full oribt?
Content:
what project covers…
focus on maybe one example
explain to audience what trying to do and what maths/computing
is better to show understanding which is key


13/3

concentrate on presentation and writing up now
should be understandable for fellow students and in own words
will know if too much like the textbook
check full obrit and gc agree
play around with dipole field example
rotating dipole
dipole on tilt. different planets
would need to solve faradays eqn
rotating dipole in litreture might be on website
not jupiter or saturn as nearly algined rotators maybe neputne uranus
mpore incluened rotation axis planets
can give draft of report once
length of write up doesn’t matter too much
appendix add code there maybe


next example rotating dipole field
rotating b field induces e field via faradays law
E produces ExB drift which velocity is omega cross r
particle is dragged along corotating with planet
in jupiters example it dominates all other drifts
baumjohann textbook page 82/83 corotation

omega cross r
omega (0,0,omega) as rotation around z
r (x,y,z)


17/3

ohms law - fluid eqn
maxwells eqn
not simple deriving
was thinking maxwell eqn approach
check both are the same
check old papers 1950s?

start E + vxB = 0
vrot = omegaxr
B(r,theta,phi-omegat) which is phi bar in diff frame of ref

calculate vector potential then take d/dt
what is the vector potential textbook or online
search vector potential for dipole field
they give everything alilgned with on eof the axus
then give tilt to regular acis
dipole moment at angle with omega with rotatuin axis

how to put tikt into code?
catesian?
xy coord t dep
sin omega t  time dep
look at how go from sphericla to catsian
xy have some r sin theta cos phi
phi bar here

that can be split into 
may have to go from
coroting frame into one that deosn’t rotate
suspisious about what textbook does although could be correct
could work it out if had time

could say:
rigidly rotating dipole
dont have to do all byself
key thing is the particle orbit theory
just make sure everything is correct
and quote source
google search first place to look

E=-dA/dt is important

take the traumann example
dissadvantage parallel E field if you put in you get accel along field lines
assume that ideal ohms law is valid 
can figure how to make it work
close to earth many planetary magentosphere
then dont have to worry about E parallel thing
hopefully wont make it too hard to find E field
call other functions

frozen condition?

particle level are stuck to the field line rotate with field

grad and curv drift
also get ExB drift
should dominate all the other drifts
bounce and gryrates while rotating with dipole field


presentation:
general overview
max 1-2 points to talk about
could be demonstration of guiding centre in dipole field?
give a table of contents at start 
what project does
beginning show something with constant field
some survive in cruved and inhomogenous gield
extra drift motions
mag momentio
guiding centre
mirror in dipole field
curvatre and grad b drift
nice to demonstrate
with figures i have 
if ready rotating dipole field
assume that plasma corrotates oartifle  with exb drift
depending how slides are
apart from tutle and summary 
so my project is about this
and figure and list of bullet points
talk about this
should add all text onto slide
speak with the slides
the best speaker just have a figure on the slide
rule is 1-2 mins per slide
main thing know what you’re talking about make slides easy to understand 
advice is dont talk about the complete project
pick about certian bits
demonstrate what the drifts do 
in dipole field
relevent to earth magnetosphere
possibility


Questions to ask next meeting