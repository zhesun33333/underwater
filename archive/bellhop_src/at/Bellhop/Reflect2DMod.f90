MODULE Reflect2DMod

  ! This is the version used by BELLHOP3D in the Nx2D case

  USE bellhopMod
  IMPLICIT NONE
CONTAINS

  SUBROUTINE Reflect2D( is, HS, BotTop, nBdry3d, z_xx, z_xy, z_yy, kappa_xx, kappa_xy, kappa_yy, RefC, Npts, tradial )

    USE RefCoef
    USE sspMod
    USE Cone   ! if using analytic formulas for the cone (conical seamount)
    USE AttenMod

    INTEGER,                INTENT( IN    ) :: Npts
    REAL     (KIND=8),      INTENT( IN    ) :: tradial( 2 )
    REAL     (KIND=8),      INTENT( IN    ) :: nBdry3d( 3 )            ! Normal to the boundary
    REAL     (KIND=8),      INTENT( IN    ) :: z_xx, z_xy, z_yy, kappa_xx, kappa_xy, kappa_yy  ! Boundary curvature
    CHARACTER (LEN=3),      INTENT( IN    ) :: BotTop                  ! Flag indicating bottom or top reflection
    TYPE( HSInfo ),         INTENT( INOUT ) :: HS                      ! Halfspace properties
    TYPE( ReflectionCoef ), INTENT( IN    ) :: RefC( NPts )            ! reflection coefficient
    INTEGER,                INTENT( INOUT ) :: is                      ! index of the ray step
    INTEGER           :: is1
    REAL     (KIND=8) :: c, cimag, gradc( 2 ), crr, crz, czz, rho                  ! derivatives of sound speed
    REAL     (KIND=8) :: Tg, Th                                                    ! components of ray tangent
    REAL     (KIND=8) :: ck, co, si, cco, ssi, pdelta, rddelta, sddelta            ! for beam shift
    COMPLEX  (KIND=8) :: gamma1, gamma2, gamma1Sq, gamma2Sq, GK, ch, a, b, d, sb, delta, ddelta
    COMPLEX  (KIND=8) :: kx, kz, kzP, kzS, kzP2, kzS2, mu, f, g, y2, y4, Refl      ! for tabulated reflection coef.
    REAL     (KIND=8) :: kappa                                                     ! Boundary curvature
    REAL     (KIND=8) :: tBdry( 2 ), nBdry( 2 )                                    ! tangent, normal to boundary
    TYPE( ReflectionCoef ) :: RInt

    is  = is + 1
    is1 = is + 1

    ! following should perhaps be pre-calculated in Bdry3DMod
    nBdry( 1 ) = DOT_PRODUCT( nBdry3d( 1 : 2 ), tradial )
    nBdry( 2 ) = nBdry3d( 3 )

    !CALL ConeFormulas( z_xx, z_xy, z_yy, nBdry, xs_3D, tradial, ray2D( is )%x, BotTop ) ! analytic formulas for the curvature of the seamount

!!!! use kappa_xx or z_xx?
    ! kappa = ( z_xx * tradial( 1 ) ** 2 + 2 * z_xy * tradial( 1 ) * tradial( 2 ) + z_yy * tradial( 2 ) ** 2 )
    kappa = ( kappa_xx * tradial( 1 ) ** 2 + 2 * kappa_xy * tradial( 1 ) * tradial( 2 ) + kappa_yy * tradial( 2 ) ** 2 )

    IF ( BotTop == 'TOP' ) kappa = -kappa

    Th = DOT_PRODUCT( ray2D( is )%t, nBdry )  ! component of ray tangent normal to boundary
    nBdry = nBdry / NORM2( nBdry )
    tBdry = ray2D( is )%t - Th * nBdry        ! component of ray tangent along the boundary
    tBdry = tBdry / NORM2( tBdry )
    ! could also calculate tbdry as +/- of [ nbdry( 2), -nbdry( 1 ) ], but need sign

    Tg = DOT_PRODUCT( ray2D( is )%t, tBdry )  ! component of ray tangent along the boundary

    CALL EvaluateSSP2D( ray2D( is1 )%x, c, cimag, gradc, crr, crz, czz, rho, xs_3D, tradial, freq )   ! just to get c

    ray2D( is1 )%NumTopBnc = ray2D( is )%NumTopBnc
    ray2D( is1 )%NumBotBnc = ray2D( is )%NumBotBnc
    ray2D( is1 )%x         = ray2D( is )%x
    ray2D( is1 )%t         = ray2D( is )%t - 2.0 * Th * nBdry   ! changing the ray direction
    ray2D( is1 )%tau       = ray2D( is )%tau
    ray2D( is1 )%c         = c

    CALL CurvatureCorrection2( ray2D( is ), ray2D( is1 ) )

    ! amplitude and phase change

    SELECT CASE ( HS%BC )
    CASE ( 'R' )                 ! rigid
       ray2D( is1 )%Amp   = ray2D( is )%Amp
       ray2D( is1 )%Phase = ray2D( is )%Phase
    CASE ( 'V' )                 ! vacuum
       ray2D( is1 )%Amp   = ray2D( is )%Amp
       ray2D( is1 )%Phase = ray2D( is )%Phase + pi
    CASE ( 'F' )                 ! file
       RInt%theta = RadDeg * ABS( ATAN2( Th, Tg ) )           ! angle of incidence (relative to normal to bathymetry)
       IF ( RInt%theta > 90 ) RInt%theta = 180. - RInt%theta  ! reflection coefficient is symmetric about 90 degrees
       CALL InterpolateReflectionCoefficient( RInt, RefC, Npts, PRTFile )
       ray2D( is1 )%Amp   = ray2D( is )%Amp   * RInt%R
       ray2D( is1 )%Phase = ray2D( is )%Phase + RInt%phi
    CASE ( 'A', 'G' )            ! half-space
       kx = omega * Tg     ! wavenumber in direction parallel      to bathymetry
       kz = omega * Th     ! wavenumber in direction perpendicular to bathymetry (in ocean)

       IF ( HS%BC == 'G' ) THEN   ! if grain-size option, calculate sound speed in sediment from ratio
          ! AttenUnit = 'L'   ! loss parameter
          ! the term vr / 1000 converts vr to units of m per ms 
          alphaR = HS%vr * c
          alphaI = HS%alpha2_f * ( HS%vr / 1000 ) * 1500.0 * log( 10.0 ) / ( 40.0 * pi )   ! loss parameter Sect. IV., Eq. (4) of handbook

          HS%cp  = CRCI( zTemp, alphaR, alphaI, freq, freq, 'L ', betaPowerLaw, ft )
          HS%cs  = 0.0
          ! WRITE( PRTFile, FMT = "( 'Converted sound speed =', 2F10.2, 3X, 'density = ', F10.2, 3X, 'loss parm = ', F10.4 )" ) &
          !     HS%cp, HS%rho, alphaI
       END IF

       ! notation below is a bit misleading
       ! kzS, kzP is really what I called gamma in other codes, and differs by a factor of +/- i
       IF ( REAL( HS%cS ) > 0.0 ) THEN   ! elastic medium
          kzS2 = kx ** 2 - ( omega / HS%cS ) ** 2
          kzP2 = kx ** 2 - ( omega / HS%cP ) ** 2
          kzS  = SQRT( kzS2 )
          kzP  = SQRT( kzP2 )
          mu   = HS%rho * HS%cS ** 2

          y2 = ( ( kzS2 + kx ** 2 ) ** 2 - 4.0D0 * kzS * kzP * kx ** 2 ) * mu
          y4 = kzP * ( kx ** 2 - kzS2 )

          f = omega ** 2 * y4
          g = y2
       ELSE
          kzP = SQRT( kx ** 2 - ( omega / HS%cP ) ** 2 )

          ! Intel and GFortran compilers return different branches of the SQRT for negative reals
          IF ( REAL( kzP ) == 0.0D0 .AND. AIMAG( kzP ) < 0.0D0 ) kzP = -kzP
          f   = kzP
          g   = HS%rho
       END IF

       Refl =  - ( rho * f - i * kz * g ) / ( rho * f + i * kz * g )   ! complex reflection coef.

       IF ( ABS( Refl ) < 1.0E-5 ) THEN   ! kill a ray that has lost its energy in reflection
          ray2D( is1 )%Amp   = 0.0
          ray2D( is1 )%Phase = ray2D( is )%Phase
       ELSE
          ray2D( is1 )%Amp   = ABS( Refl ) * ray2D(  is )%Amp
          ray2D( is1 )%Phase = ray2D( is )%Phase + ATAN2( AIMAG( Refl ), REAL( Refl ) )

          ! compute beam-displacement Tindle, Eq. (14)
          ! needs a correction to beam-width as well ...
          !  IF ( REAL( gamma2Sq ) < 0.0 ) THEN
          !     rhoW   = 1.0   ! density of water
          !     rhoWSq  = rhoW  * rhoW
          !     rhoHSSq = rhoHS * rhoHS
          !     DELTA = 2 * GK * rhoW * rhoHS * ( gamma1Sq - gamma2Sq ) /
          ! &( gamma1 * i * gamma2 *
          ! &( -rhoWSq * gamma2Sq + rhoHSSq * gamma1Sq ) )
          !     RV( is + 1 ) = RV( is + 1 ) + DELTA
          !  END IF

          if ( Beam%Type( 4 : 4 ) == 'S' ) then   ! beam displacement & width change (Seongil's version)
             ch = ray2D( is )%c / conjg( HS%cP )
             co = ray2D( is )%t( 1 ) * ray2D( is )%c
             si = ray2D( is )%t( 2 ) * ray2D( is )%c
             ck = omega / ray2D( is )%c

             a   = 2 * HS%rho * ( 1 - ch * ch )
             b   = co * co - ch * ch
             d   = HS%rho * HS%rho * si * si + b
             sb  = sqrt( b )
             cco = co * co
             ssi = si * si

             IF ( si /= 0.0 ) THEN
                delta = a * co / si / ( ck * sb * d )   ! Do we need an abs() on this???
             ELSE
                delta = 0.0
             END IF

             pdelta  = real( delta ) / ( ray2D( is )%c / co)
             ddelta  = -a / ( ck*sb*d ) - a*cco / ssi / (ck*sb*d) + a*cco / (ck*b*sb*d) &
                  -a*co / si / (ck*sb*d*d) * (2* HS%rho * HS%rho *si*co-2*co*si)
             rddelta = -real( ddelta )
             sddelta = rddelta / abs( rddelta )        

             print *, 'beam displacing ...'
             ray2D( is1 )%x( 1 ) = ray2D( is1 )%x( 1 ) + real( delta )                           ! displacement
             ray2D( is1 )%tau    = ray2D( is1 )%tau + pdelta                                     ! phase change
             ray2D( is1 )%q      = ray2D( is1 )%q + sddelta * rddelta * si * c * ray2D( is )%p   ! beam-width change
          end if

       END IF
    CASE DEFAULT
       WRITE( PRTFile, * ) 'HS%BC = ', HS%BC
       CALL ERROUT( 'Reflect2D', 'Unknown boundary condition type' )
    END SELECT

  CONTAINS

    SUBROUTINE CurvatureCorrection2( ray, rayOut )

      TYPE( ray2DPt ), INTENT( IN    ) :: ray
      TYPE( ray2DPt ), INTENT( INOUT ) :: rayOut
      REAL (KIND=8) :: RM, RN                        ! curvature corrections on reflection
      REAL (KIND=8) :: urayt( 2 ), urayn( 2 ), uraytOut( 2 ), uraynOut( 2 ), cnjump, csjump  ! for curvature change

      ! Calculate the change in curvature
      ! Based on formulas given by Muller, Geoph. J. R.A.S., 79 (1984).

      ! incident unit ray tangent and normal
      urayt = c * ray%t                               ! unit tangent to ray
      urayn = [ -urayt( 2 ), urayt( 1 ) ]             ! unit normal  to ray

      ! reflected unit ray tangent and normal (the reflected tangent, normal system has a different orientation)
      uraytOut = c * rayOut%t                         ! unit tangent to ray
      uraynOut = -[ -uraytOut( 2 ), uraytOut( 1 ) ]   ! unit normal  to ray

      RN = 2 * kappa / c ** 2 / Th    ! boundary curvature correction

      ! get the jumps (this could be simplified, e.g. jump in rayt is roughly 2 * Th * nbdry
      cnjump = -DOT_PRODUCT( gradc, uraynOut - urayn )
      csjump = -DOT_PRODUCT( gradc, uraytOut - urayt )

      IF ( BotTop == 'TOP' ) THEN
         cnjump = -cnjump   ! this is because the (t,n) system of the top boundary has a different sense to the bottom boundary
         RN     = -RN
      END IF

      RM = Tg / Th   ! this is tan( alpha ) where alpha is the angle of incidence
      RN = RN + RM * ( 2 * cnjump - RM * csjump ) / c ** 2

      SELECT CASE ( Beam%Type( 3 : 3 ) )
      CASE ( 'D' )
         RN = 2.0 * RN
      CASE ( 'Z' )
         RN = 0.0
      END SELECT

      rayOut%p   = ray%p + ray%q * RN
      rayOut%q   = ray%q

    END SUBROUTINE CurvatureCorrection2

  END SUBROUTINE Reflect2D

END MODULE Reflect2DMod
