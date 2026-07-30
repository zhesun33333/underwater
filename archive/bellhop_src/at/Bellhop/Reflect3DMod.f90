MODULE Reflect3DMod

  ! This is the version used by BELLHOP3D in the 3D case

  USE bellhopMod
  IMPLICIT NONE
CONTAINS

  SUBROUTINE Reflect3D( is, HS, BotTop, nBdry, z_xx, z_xy, z_yy, kappa_xx, kappa_xy, kappa_yy, RefC, Npts )

    USE RefCoef
    USE sspMod
    USE AttenMod

    INTEGER,                INTENT( IN    ) :: Npts                    ! Number of points in the reflection coefficient
    REAL     (KIND=8),      INTENT( INOUT ) :: nBdry( 3 )              ! Normal to the boundary (changes if cone reflection)
    REAL     (KIND=8),      INTENT( IN    ) :: z_xx, z_xy, z_yy, kappa_xx, kappa_xy, kappa_yy  ! Boundary curvature
    CHARACTER (LEN=3),      INTENT( IN    ) :: BotTop                  ! Flag indicating bottom or top reflection
    TYPE( HSInfo ),         INTENT( INOUT ) :: HS                      ! Halfspace properties (calculated if grain size given in sediment)
    TYPE( ReflectionCoef ), INTENT( IN    ) :: RefC( NPts )            ! reflection coefficient
    INTEGER,                INTENT( INOUT ) :: is                      ! index of the ray step
    INTEGER           :: is1
    REAL     (KIND=8) :: c, cimag, gradc( 3 ), cxx, cyy, czz, cxy, cxz, cyz, rho   ! derivatives of sound speed
    REAL     (KIND=8) :: Tg, Th                                                    ! components of ray tangent
    COMPLEX  (KIND=8) :: kx, kz, kzP, kzS, kzP2, kzS2, mu, f, g, y2, y4, Refl      ! for tabulated reflection coef.
    TYPE( ReflectionCoef ) :: RInt
    REAL     (KIND=8) :: tBdry( 3 )                                                ! tangent to the boundary
    REAL     (KIND=8) :: kappaMat( 2, 2 )                                          ! Boundary curvature

    is  = is + 1
    is1 = is + 1

    !CALL ConeFormulas(    z_xx, z_xy, z_yy, nBdry, xs, ray3D( is )%x, BotTop ) ! analytic formulas for the curvature of the seamount
    !CALL ParabotFormulas( z_xx, z_xy, z_yy, nBdry,     ray3D( is )%x, BotTop ) ! analytic formulas for the curvature of the parabolic bottom
    !write( *, * ) 'z_xx, z_xy, z_yy', z_xx, z_xy, z_yy, 'nBdry', nBdry
    kappaMat( 1, 1 ) = z_xx / 2
    kappaMat( 1, 2 ) = z_xy / 2
    kappaMat( 2, 1 ) = z_xy / 2
    kappaMat( 2, 2 ) = z_yy / 2

    ! get normal and tangential components of ray in the reflecting plane

    Th = DOT_PRODUCT( ray3D( is )%t, nBdry )  ! component of ray tangent normal to boundary
    tBdry = ray3D( is )%t - Th * nBdry        ! component of ray tangent along the boundary, in the reflection plane
    tBdry = tBdry / NORM2( tBdry )
    Tg = DOT_PRODUCT( ray3D( is )%t, tBdry )  ! component of ray tangent along the boundary

    CALL EvaluateSSP3D( ray3D( is1 )%x, c, cimag, gradc, cxx, cyy, czz, cxy, cxz, cyz, rho, freq, 'TAB' )

    ray3D( is1 )%NumTopBnc = ray3D( is )%NumTopBnc
    ray3D( is1 )%NumBotBnc = ray3D( is )%NumBotBnc
    ray3D( is1 )%x         = ray3D( is )%x
    ray3D( is1 )%t         = ray3D( is )%t - 2.0 * Th * nBdry   ! changing the ray direction
    ray3D( is1 )%tau       = ray3D( is )%tau
    ray3D( is1 )%c         = c

    CALL CurvatureCorrection2( ray3D( is ), ray3D( is1 ) )

    ! amplitude and phase change

    SELECT CASE ( HS%BC )
    CASE ( 'R' )                 ! rigid
       ray3D( is1 )%Amp   = ray3D( is )%Amp
       ray3D( is1 )%Phase = ray3D( is )%Phase
    CASE ( 'V' )                 ! vacuum
       ray3D( is1 )%Amp   = ray3D( is )%Amp
       ray3D( is1 )%Phase = ray3D( is )%Phase + pi
    CASE ( 'F' )                 ! file
       RInt%theta = RadDeg * ABS( ATAN2( Th, Tg ) )           ! angle of incidence (relative to normal to bathymetry)
       IF ( RInt%theta > 90 ) RInt%theta = 180. - RInt%theta  ! reflection coefficient is symmetric about 90 degrees
       CALL InterpolateReflectionCoefficient( RInt, RefC, Npts, PRTFile )
       ray3D( is1 )%Amp   = ray3D( is )%Amp   * RInt%R
       ray3D( is1 )%Phase = ray3D( is )%Phase + RInt%phi
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

       ! write( *, * ) abs( Refl ), c, HS%cp, rho, HS%rho       
       ! Hack to make a wall (where the bottom slope is more than 80 degrees) be a perfect reflector
       !IF ( ABS( RadDeg * ATAN2( nBdry( 3 ), NORM2( nBdry( 1 : 2 ) ) ) ) < 0 ) THEN   ! was 60 degrees
       !   Refl = 1
       !END IF

       IF ( ABS( Refl ) < 1.0E-5 ) THEN   ! kill a ray that has lost its energy in reflection
          ray3D( is1 )%Amp   = 0.0
          ray3D( is1 )%Phase = ray3D( is )%Phase
       ELSE
          ray3D( is1 )%Amp   = ABS( Refl ) * ray3D(  is )%Amp
          ray3D( is1 )%Phase = ray3D( is )%Phase + ATAN2( AIMAG( Refl ), REAL( Refl ) )
       END IF
    CASE DEFAULT
       WRITE( PRTFile, * ) 'HS%BC = ', HS%BC
       CALL ERROUT( 'Reflect2D', 'Unknown boundary condition type' )
    END SELECT

  CONTAINS

    SUBROUTINE CurvatureCorrection2( ray, rayOut )

      USE RayNormals
      TYPE( ray3DPt ), INTENT( IN  ) :: ray
      TYPE( ray3DPt ), INTENT( OUT ) :: rayOut
      REAL (KIND=8) :: e1( 3 ), e2( 3 ), &                   ! ray normals for ray-centered coordinates
                       RM, R1, R2, R3                        ! curvature corrections on reflection
      REAL (KIND=8) :: p_tilde_in(  2 ), p_hat_in(  2 ), q_tilde_in(  2 ), q_hat_in(  2 ), p_tilde_out( 2 ), p_hat_out( 2 )
      REAL (KIND=8) :: cn1jump, cn2jump, csjump 
      REAL (KIND=8) :: RotMat( 2, 2 ), DMat( 2, 2 ), DMatTemp( 2, 2 )
      REAL (KIND=8) :: rayt( 3 ), rayn1( 3 ), rayn2( 3 )             ! unit ray tangent and normals
      REAL (KIND=8) :: rayt_tilde( 3 ), rayn1_tilde( 3 ), rayn2_tilde( 3 )
      REAL (KIND=8) :: t_rot( 2 ), n_rot( 2 )

      ! Calculate the ray normals, rayn1, rayn2, and a unit tangent
      
      CALL CalcTangent_Normals( ray%t,    nBdry, rayt,       rayn1,       rayn2       ) ! incident
      CALL CalcTangent_Normals( rayOut%t, nBdry, rayt_tilde, rayn1_tilde, rayn2_tilde ) ! reflected

      ! rotation matrix to get surface curvature in and perpendicular to the reflection plane
      ! we use only the first two elements of the vectors because we want the projection in the x-y plane
      t_rot = rayt(  1 : 2 ) / NORM2( rayt(  1 : 2 ) )
      n_rot = rayn2( 1 : 2 ) / NORM2( rayn2( 1 : 2 ) )

      RotMat( 1 : 2, 1 ) = t_rot
      RotMat( 1 : 2, 2 ) = n_rot

      ! apply the rotation to get the matrix D of curvatures (see Popov 1977 for definition of DMat)
      ! DMat = RotMat^T * kappaMat * RotMat, with RotMat anti-symmetric
      DMatTemp = MATMUL( 2 * kappaMat, RotMat )
      DMat     = MATMUL( TRANSPOSE( RotMat ), DMatTemp )

      ! normal and tangential derivatives of the sound speed
      cn1jump =  DOT_PRODUCT( gradc, -rayn1_tilde - rayn1 )
      cn2jump =  DOT_PRODUCT( gradc, -rayn2_tilde - rayn2 )
      csjump  = -DOT_PRODUCT( gradc,  rayt_tilde  - rayt  )

!!! not sure if cn2 needs a sign flip also
!!$  IF ( BotTop == 'TOP' ) THEN
!!$     cn1jump = -cn1jump    ! flip sign for top reflection
!!$     cn2jump = -cn2jump    ! flip sign for top reflection
!!$  END IF

      ! Note that Tg, Th need to be multiplied by c to normalize tangent; hence, c^2 below
      ! Added the SIGN in R2 to make ati and bty have a symmetric effect on the beam
      ! Not clear why that's needed
      
      RM = Tg / Th   ! this is tan( alpha ) where alpha is the angle of incidence
      R1 = 2 / c ** 2 * DMat( 1, 1 ) / Th + RM * ( 2 * cn1jump - RM * csjump ) / c ** 2
      R2 = 2 / c *      DMat( 1, 2 ) * SIGN( 1.0D0, -Th )      + RM * cn2jump  / c ** 2
      R3 = 2 *          DMat( 2, 2 ) * Th

      ! z-component of unit tangent is sin( theta ); we want cos( theta )
      ! R1 = R1 * ( 1 - ( ray%c * ray%t( 3 ) ) ** 2 )
      ! write( *, * )  1 - ( ray%c * ray%t( 3 ) ) ** 2

      ! *** curvature correction ***

      CALL RayNormal( ray%t, ray%phi, ray%c, e1, e2 )  ! Compute ray normals e1 and e2

      ! rotate p-q from e1, e2 system, onto rayn1, rayn2 system

      RotMat( 1, 1 ) = DOT_PRODUCT( rayn1, e1 )
      RotMat( 1, 2 ) = DOT_PRODUCT( rayn1, e2 )
      RotMat( 2, 1 ) = -RotMat( 1, 2 )             ! same as DOT_PRODUCT( rayn2, e1 )
      RotMat( 2, 2 ) = DOT_PRODUCT( rayn2, e2 )

      p_tilde_in = RotMat( 1, 1 ) * ray%p_tilde + RotMat( 1, 2 ) * ray%p_hat
      p_hat_in   = RotMat( 2, 1 ) * ray%p_tilde + RotMat( 2, 2 ) * ray%p_hat

      q_tilde_in = RotMat( 1, 1 ) * ray%q_tilde + RotMat( 1, 2 ) * ray%q_hat
      q_hat_in   = RotMat( 2, 1 ) * ray%q_tilde + RotMat( 2, 2 ) * ray%q_hat

      ! here's the actual curvature change

      p_tilde_out = p_tilde_in + q_tilde_in * R1 - q_hat_in * R2
      p_hat_out   = p_hat_in   + q_tilde_in * R2 + q_hat_in * R3
      !p_hat_out   = p_hat_in   + q_tilde_in * R2 - q_hat_in * R3 ! this one good

      ! rotate p-q back to e1, e2 system (RotMat^(-1) = RotMat^T)

      rayOut%p_tilde = RotMat( 1, 1 ) * p_tilde_out + RotMat( 2, 1 ) * p_hat_out
      rayOut%p_hat   = RotMat( 1, 2 ) * p_tilde_out + RotMat( 2, 2 ) * p_hat_out

      rayOut%q_tilde = ray%q_tilde
      rayOut%q_hat   = ray%q_hat

      ! write( *, * ) 'p', ray%p_tilde, ray%p_hat, rayOut%p_tilde, rayOut%p_hat
      ! write( *, * ) 'q', ray%q_tilde, ray%q_hat, rayOut%q_tilde, rayOut%q_hat

      ! Logic below fixes a bug when the |dot product| is infinitesimally greater than 1 (then ACos is complex)
      !rayOut%phi = ray%phi + 2.0D0 * ACOS( MAX( MIN( DOT_PRODUCT( rayn1, e1 ), 1.0D0 ), -1.0D0 ) ) !!!What happens to torsion?
      rayOut%phi = ray%phi
      
      ! f, g, h continuation; needs curvature corrections
      ! ray3D( is1 )%f    = ray3D( is )%f ! + ray3D( is )%DetQ * R1
      ! ray3D( is1 )%g    = ray3D( is )%g ! + ray3D( is )%DetQ * R1
      ! ray3D( is1 )%h    = ray3D( is )%h ! + ray3D( is )%DetQ * R2
      ! ray3D( is1 )%DetP = ray3D( is )%DetP
      ! ray3D( is1 )%DetQ = ray3D( is )%DetQ

    END SUBROUTINE CurvatureCorrection2

    ! ********************************************************************** !

    SUBROUTINE CalcTangent_Normals( ray3Dt, nBdry, rayt, rayn1, rayn2 )

      ! given the ray tangent vector (not normalized) and the unit normal to the bdry
      ! calculate a unit tangent and the two ray normals

      USE cross_products

      REAL (KIND=8), INTENT( IN )  :: ray3Dt( 3 ), nBdry( 3 )
      REAL (KIND=8), INTENT( OUT ) :: rayt( 3 ), rayn1( 3 ), rayn2( 3 )

      ! incident unit ray tangent and normal
      rayt  = c * ray3Dt                      ! unit tangent to ray
      rayn2 = -cross_product( rayt, nBdry )   ! ray tangent x boundary normal gives refl. plane normal
      rayn2 = rayn2 / NORM2( rayn2 )          ! unit normal to refl. plane
      rayn1 = -cross_product( rayt, rayn2 )   ! ray tangent x refl. plane normal is first ray normal

    END SUBROUTINE CalcTangent_Normals

  END SUBROUTINE Reflect3D

END MODULE Reflect3DMod
