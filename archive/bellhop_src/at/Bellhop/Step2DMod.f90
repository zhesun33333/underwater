MODULE Step2DMod

  USE bellhopMod
  USE Bdry3DMod
  USE sspMod
  IMPLICIT NONE
CONTAINS

  SUBROUTINE Step2D( ray0, ray2, tradial )

    ! Does a single step along the ray
    ! x denotes the ray coordinate, ( r, z )
    ! t denotes the scaled tangent to the ray (previously ( rho, zeta ) )
    ! c * t would be the unit tangent

    USE Step3DMod
    TYPE( ray2DPt ), INTENT( IN  ) :: ray0
    TYPE( ray2DPt )                :: ray1
    TYPE( ray2DPt ), INTENT( OUT ) :: ray2
    REAL ( KIND=8 ), INTENT( IN  ) :: tradial( 2 )   ! coordinate of source and ray bearing angle
    INTEGER         :: iSegx0, iSegy0, iSegz0
    REAL ( KIND=8 ) :: gradc0( 2 ), gradc1( 2 ), gradc2( 2 ), rho, &
         c0, cimag0, csq0, crr0, crz0, czz0, cnn0_csq0, &
         c1, cimag1, csq1, crr1, crz1, czz1, cnn1_csq1, &
         c2, cimag2,       crr2, crz2, czz2, &
         urayt0( 2 ), urayt1( 2 ), &
         h, halfh, hw0, hw1, w0, w1, RM, RN, &
         ray2n( 2 ), gradcjump( 2 ), cnjump, csjump, &
         rayx3D( 3 ), rayt3D( 3 ) 

    ! write( *, * ) ray0%x, ray0%p, ray0%q

    ! The numerical integrator used here is a version of the polygon (a.k.a. midpoint, leapfrog, or Box method), and similar
    ! to the Heun (second order Runge-Kutta method).
    ! However, it's modified to allow for a dynamic step change, while preserving the second-order accuracy).

    ! *** Phase 1 (an Euler step)

    CALL EvaluateSSP2D( ray0%x, c0, cimag0, gradc0, crr0, crz0, czz0, rho, xs_3D, tradial, freq )

    csq0      = c0 * c0
    cnn0_csq0 = crr0 * ray0%t( 2 )**2 - 2.0 * crz0 * ray0%t( 1 ) * ray0%t( 2 ) + czz0 * ray0%t( 1 )**2
    urayt0    = c0 * ray0%t   ! unit tangent to ray

    iSegx0 = iSegx            ! make note of current layer
    iSegy0 = iSegy
    iSegz0 = iSegz

    h = Beam%deltas           ! initially set the step h, to the basic one, deltas

    rayx3D = [ xs_3D( 1 ) + ray0%x( 1 ) * tradial( 1 ), xs_3D( 2 ) + ray0%x( 1 ) * tradial( 2 ), ray0%x( 2 ) ]
    rayt3D = [              urayt0( 1 ) * tradial( 1 ),              urayt0( 1 ) * tradial( 2 ), urayt0( 2 ) ]

    CALL ReduceStep3D( rayx3D, rayt3D, iSegx0, iSegy0, iSegz0, h ) ! reduce h to land on boundary
    halfh = 0.5 * h   ! first step of the modified polygon method is a half step

    ray1%x = ray0%x + halfh * urayt0
    ray1%t = ray0%t - halfh * gradc0 / csq0
    ray1%p = ray0%p - halfh * cnn0_csq0 * ray0%q
    ray1%q = ray0%q + halfh * c0        * ray0%p

    ! write( *, * ) ray1%x, ray1%p, ray1%q

    ! *** Phase 2

    CALL EvaluateSSP2D( ray1%x, c1, cimag1, gradc1, crr1, crz1, czz1, rho, xs_3D, tradial, freq )
    csq1      = c1 * c1
    cnn1_csq1 = crr1 * ray1%t( 2 )**2 - 2.0 * crz1 * ray1%t( 1 ) * ray1%t( 2 ) + czz1 * ray1%t( 1 )**2
    urayt1    = c1 * ray1%t   ! unit tangent to ray

    ! The Munk test case with a horizontally launched ray caused problems.
    ! The ray vertexes on an interface and can ping-pong around that interface.
    ! Have to be careful in that case about big changes to the stepsize (that invalidate the leap-frog scheme) in phase II.
    ! A modified Heun or Box method could also work.

    rayt3D = [              urayt1( 1 ) * tradial( 1 ),              urayt1( 1 ) * tradial( 2 ), urayt1( 2 ) ]

    CALL ReduceStep3D( rayx3D, rayt3D, iSegx0, iSegy0, iSegz0, h ) ! reduce h to land on boundary

    ! use blend of f' based on proportion of a full step used.
    w1  = h / ( 2.0d0 * halfh )
    w0  = 1.0d0 - w1
    hw0 = h * w0
    hw1 = h * w1

    ray2%x   = ray0%x   + hw0 * urayt0                      + hw1 * urayt1
    ray2%t   = ray0%t   - hw0 * gradc0 / csq0               - hw1 * gradc1 / csq1
    ray2%tau = ray0%tau + hw0 / CMPLX( c0, cimag0, KIND=8 ) + hw1 / CMPLX( c1, cimag1, KIND=8 )
    ray2%p   = ray0%p   - hw0 * cnn0_csq0 * ray0%q          - hw1 * cnn1_csq1 * ray1%q
    ray2%q   = ray0%q   + hw0 * c0        * ray0%p          + hw1 * c1        * ray1%p

    ray2%Amp       = ray0%Amp
    ray2%Phase     = ray0%Phase
    ray2%NumTopBnc = ray0%NumTopBnc
    ray2%NumBotBnc = ray0%NumBotBnc

    ! If we crossed an interface, apply jump condition

    CALL EvaluateSSP2D( ray2%x, c2, cimag2, gradc2, crr2, crz2, czz2, rho, xs_3D, tradial, freq )
    ray2%c = c2

!!!!!! this needs modifying like the full 3D version to handle jumps in the x-y direction
    IF ( iSegx /= iSegx0 .OR. &
         iSegy /= iSegy0 .OR. &
         iSegz /= iSegz0 ) THEN

       SELECT CASE ( SSP%Type )     ! is their a discontinuity in the first derivative?
       CASE ( 'N', 'C', 'Q', 'H' )  ! N2-linear or C-linear profile option
          gradcjump =  gradc2 - gradc0  ! this is precise only for c-linear layers
          ray2n     = [ -ray2%t( 2 ), ray2%t( 1 ) ]   ! ray normal

          cnjump    = DOT_PRODUCT( gradcjump, ray2n  )
          csjump    = DOT_PRODUCT( gradcjump, ray2%t )

          RM     = ray2%t( 1 ) / ray2%t( 2 )
          RN     = RM * ( 2 * cnjump - RM * csjump ) / c2
          RN     = -RN
          ray2%p = ray2%p + ray2%q * RN
       CASE ( 'P', 'S' )            ! monotone PCHIP ACS or Cubic spline profile option
          gradcjump = 0.0
       CASE ( 'A' )                 ! Analytic profile option
          gradcjump = 0.0
       CASE DEFAULT
          WRITE( PRTFile, * ) 'Profile option: ', SSP%Type
          CALL ERROUT( 'BELLHOP: Step2D', 'Invalid profile option' )
       END SELECT

    END IF

    ! write( *, * ) ray2%x, ray2%p, ray2%q

  END SUBROUTINE Step2D

END MODULE Step2DMod
