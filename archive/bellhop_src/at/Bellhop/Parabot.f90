
MODULE Parabot
  USE bellhopMod
  USE MathConstants
  IMPLICIT NONE

CONTAINS

    SUBROUTINE ParabotFormulas( z_xx, z_xy, z_yy, nBdry, xray, BotTop )

      ! analytic formula for the parabolic bottom

      REAL     (KIND=8), INTENT( IN  ) :: xray( 3 )
      CHARACTER (LEN=3), INTENT( IN  ) :: BotTop                  ! Flag indicating bottom or top reflection
      REAL     (KIND=8), INTENT( OUT ) :: z_xx, z_xy, z_yy, nBdry( 3 )
      REAL (KIND=8) :: Radius, x, y, z, z_r, z_rr, z_x, z_y, a, b, c, RLen

      IF ( BotTop == 'BOT' ) THEN
         a = 1 / 1000.
         b = 0
         c = 250

         x = xray( 1 )
         y = xray( 2 )

         Radius = SQRT( x ** 2 + y ** 2 )   ! radius of seamount at the bounce point

         z = 500.0 * SQRT( 1 + Radius / c )

         z_r = 1 / ( 2 * a * z + b )
         z_x = z_r * x / Radius
         z_y = z_r * y / Radius

         nBdry = [ -z_x, -z_y, 1.D0 ]       !     normal to bdry (outward pointing)
         RLen = NORM2( nBdry )
         nBdry = nBdry / Rlen   ! unit normal to bdry
         ! curvatures

         ! z = z_xx / 2 * x^2 + z_xy * xy + z_yy * y^2 ! coefs are the kappa matrix
         ! RLen is an extra factor based on Eq. 4.4.18 in Cerveny's book
         ! It represents a length along the tangent for a delta x step

         z_rr = -1 / ( 4 * a * a * z * z * z )
         z_xx = ( z_rr * x * x / Radius ** 2 + z_r * y * y / Radius ** 3 ) / RLen
         z_xy = ( z_rr * x * y / Radius ** 2 - z_r * x * y / Radius ** 3 ) / RLen
         z_yy = ( z_rr * y * y / Radius ** 2 + z_r * x * x / Radius ** 3 ) / RLen

         ! write( *, * ) 'x=', x, 'y=', y, 'z_x', z_x, 'z_y', z_y, 'z_r', z_r, 'zrr', z_rr, 'Radius', Radius, 'RLen', RLen 
      END IF

    END SUBROUTINE ParabotFormulas

  END MODULE Parabot

