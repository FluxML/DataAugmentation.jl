# [`ComposedProjectiveTransform`](@ref) implements efficient composition
# of `ProjectiveTransform`s.

"""
    ComposedProjectiveTransform(tfms...)

Wrap multiple projective `tfms` and apply them efficiently.
The projections are fused into a single projection and only
points inside the final crop are evaluated.
"""
struct ComposedProjectiveTransform{T<:Tuple} <: ProjectiveTransform
    tfms::T
end

ComposedProjectiveTransform(tfms...) =
    ComposedProjectiveTransform(tfms)

# Composing `ProjectiveTransform`s should create a `ComposedProjectiveTransform`:

compose(tfm1::ProjectiveTransform, tfm2::ProjectiveTransform) =
    ComposedProjectiveTransform(tfm1, tfm2)

compose(composed::ComposedProjectiveTransform, tfm::ProjectiveTransform) =
    ComposedProjectiveTransform(composed.tfms..., tfm)

compose(tfm::ProjectiveTransform, composed::ComposedProjectiveTransform) =
    ComposedProjectiveTransform(tfm, composed.tfms...)


# The random state is collected from the transformations that make up the
# `ComposedProjectiveTransform`:

getrandstate(tfm::ComposedProjectiveTransform) = getrandstate.(tfm.tfms)

# To combine the concrete projections `P`, we use `ImageTransformations.compose`
# (aliased to `∘`). The bounds are also transformed at each step.


function getprojection(
        composed::ComposedProjectiveTransform,
        bounds;
        randstate = getrandstate(composed))
    P = CoordinateTransformations.IdentityTransformation()
    for (tfm, r) in zip(composed.tfms, randstate)
        P_tfm = getprojection(tfm, bounds; randstate = r)
        bounds = projectionbounds(tfm, P_tfm, bounds; randstate = r)
        P = P_tfm ∘ P
    end
    return P
end

function projectionbounds(composed::ComposedProjectiveTransform, P, bounds; randstate = getrandstate(composed))
    return transformbounds(bounds, P)
end
